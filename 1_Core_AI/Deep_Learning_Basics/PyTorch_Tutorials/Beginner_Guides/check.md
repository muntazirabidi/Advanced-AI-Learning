# UMAP Visualization of Fashion MNIST Features

In this notebook, we'll visualize how our CNN model "sees" the Fashion MNIST dataset by extracting features from the model's layers and using UMAP to reduce their dimensionality for visualization. This will help us understand what patterns the model has learned and how it distinguishes between different clothing items.

First, let's import our required libraries:

```python
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
```

Now let's load our trained model and the test dataset:

```python
# Load the model
model = FashionClassifier()
model.load_state_dict(torch.load('models/fashion_mnist_model.pth'))
model.eval()  # Set to evaluation mode

# Prepare the data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_dataset = datasets.FashionMNIST(
    root='data',
    train=False,
    download=True,
    transform=transform
)

test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
```

Let's create a function to extract features from our model. We'll get features from the layer just before the final classification layer:

```python
def extract_features(model, dataloader):
    features = []
    labels = []
    
    # Create a new model that only goes up to the feature layer
    class FeatureExtractor(nn.Module):
        def __init__(self, original_model):
            super().__init__()
            self.features = nn.Sequential(*list(original_model.children())[:-1])
            
        def forward(self, x):
            x = self.features(x)
            return x.view(x.size(0), -1)
    
    feature_extractor = FeatureExtractor(model)
    
    with torch.no_grad():
        for images, target in dataloader:
            batch_features = feature_extractor(images)
            features.append(batch_features.numpy())
            labels.append(target.numpy())
    
    return np.vstack(features), np.concatenate(labels)

# Extract features
features, labels = extract_features(model, test_loader)
```

Now let's use UMAP to reduce the dimensionality of our features. UMAP is particularly good at preserving both local and global structure in the data:

```python
# Create and fit UMAP
reducer = umap.UMAP(
    n_neighbors=15,    # Controls how local/global the embedding is
    min_dist=0.1,      # Controls how tightly points are packed
    n_components=2,    # Output dimensionality
    random_state=42    # For reproducibility
)

# Reduce dimensionality
embedding = reducer.fit_transform(features)
```

Finally, let's create a beautiful visualization:

```python
# Set up the plot style
plt.figure(figsize=(12, 8))
plt.style.use('seaborn')

# Define class names and colors
class_names = ['T-shirt/Top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']
colors = sns.color_palette('husl', n_colors=10)

# Create scatter plot
for i, label in enumerate(class_names):
    mask = labels == i
    plt.scatter(
        embedding[mask, 0],
        embedding[mask, 1],
        c=[colors[i]],
        label=label,
        alpha=0.6,
        s=50
    )

plt.title('UMAP Visualization of Fashion MNIST Features', size=14, pad=20)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
plt.tight_layout()
plt.show()

# Let's also create an interactive version using plotly
import plotly.express as px
import pandas as pd

# Create a DataFrame for plotly
df = pd.DataFrame({
    'UMAP1': embedding[:, 0],
    'UMAP2': embedding[:, 1],
    'Category': [class_names[l] for l in labels]
})

fig = px.scatter(
    df,
    x='UMAP1',
    y='UMAP2',
    color='Category',
    title='Interactive UMAP Visualization of Fashion MNIST Features',
    template='plotly_white',
    hover_data=['Category']
)

fig.update_traces(marker=dict(size=8))
fig.show()
```

Let's analyze what we can learn from these visualizations:

1. Cluster Separation: Look for clear separations between different clothing categories. Well-separated clusters indicate that our model has learned distinctive features for those categories.

2. Overlapping Categories: Notice which categories tend to overlap. These are likely the ones our model sometimes confuses.

3. Subclusters: Sometimes we might see that a single category forms multiple clusters, suggesting that our model has learned to recognize different "styles" within that category.

We can also create additional visualizations to explore specific aspects:

```python
# Visualize distance relationships
plt.figure(figsize=(12, 8))
for i in range(len(class_names)):
    mask = labels == i
    centroid = embedding[mask].mean(axis=0)
    distances = np.linalg.norm(embedding[mask] - centroid, axis=1)
    
    plt.hist(distances, bins=30, alpha=0.3, label=class_names[i])

plt.title('Distribution of Distances from Class Centroids')
plt.xlabel('Distance from Centroid')
plt.ylabel('Count')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
```

This visualization helps us understand how "tight" each class cluster is, which can indicate how confident our model is in its classifications for different categories.

The UMAP visualization provides several insights about our model:
1. How well it separates different clothing categories
2. Which categories are most similar in the model's "understanding"
3. Whether there are any interesting subgroups within categories
4. How the model's feature space is organized

Would you like to explore any particular aspect of these visualizations in more detail? We could:
1. Analyze specific clusters more closely
2. Create additional visualizations for particular categories
3. Compare UMAP with other dimensionality reduction techniques like t-SNE
4. Explore features from different layers of the model