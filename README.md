# Unsupervised Classification of Hubble Images

This code will read in a set of images, create cutouts, calculate "fingerprints" for each cutout and then do similarity data reduction.

## Processing

The general idea is given a Hubble image (for example, a full-view thumbnail), create a cutout of it. Then pass the cutout through a pre-trained neural network to create a feature vector for the cutout.  The pre-trained network used here are based on ones in Keras that are trained on the ImageNet dataset.  Then, given all the feature vectors, use the tSNE data reduction method (or similar) to reduce the dimenionsality to a 2D dataset. The similarity of the dimensionaliy reduced data is used as a surrogate for the similarity of the cutouts. 

### Simple example of the code

```
import pickle

from elodin_nn.cutout.generators import BasicCutoutGenerator
from elodin_nn.data import Data, DataCollection
from elodin_nn.fingerprint.processing import FingerprintCalculatorResnet
from elodin_nn.fingerprint.processing import calculate as fingerprint_calculate
from elodin_nn.similarity.similarity import calculate as similarity_calculate

fc_save = FingerprintCalculatorResnet().save()

#
# Load the data
#

print('Going to load the carina data')
image_data = Data(location='../../data/carina.tiff', radec=(10.7502222, -59.8677778),
                  meta={}, processing=[])
image_data.get_data()

#
# Add to the data collection
#

dc = DataCollection()
dc.add(image_data)

#
#  Create the sliding window cutout generator.
#

print('Creating cutout generator')
sliding_window_cutouts = BasicCutoutGenerator(output_size=224,
                                              step_size=100)

#
#  Create the cutouts using a sliding window cutout generator.
#

print('Creating cutouts')
cutouts = sliding_window_cutouts.create_cutouts(image_data)

#
#  Compute the fingerprints for each cutout
#

print('Calculate the fingerprint for each cutout')
fingerprints = fingerprint_calculate(cutouts, fc_save)

#
#  Compute the similarity metrics
#

print('Calculating the tSNE similarity')
similarity_tsne = similarity_calculate(fingerprints, 'tsne')

#
# Save the data to a pickle file.
#

with open('similarity_tsne.pck', 'wb') as fp:
    pickle.dump(similarity_tsne.save(), fp)
```


## Display and Interaction

Given a set of processed data there is a display method that will show the similarity plot and N similar images.  Each point in the tSNE plot is a cutout and one can click on the point in the tSNE plot. Then the 9 similar images (though configurable) will be displayed to the right along with their sky location if that is available. 

### Hubble ACS Thumbnail Images
![](https://github.com/brechmos/elodin-nn/raw/master/images/hubble_thumbnails.jpeg)

### Hubble Carina Image
![](https://github.com/brechmos/elodin-nn/raw/master/images/hubble_carina.jpeg)

### Hubble Heritage Images
![](https://github.com/brechmos/elodin-nn/raw/master/images/hubble_heritage.jpeg)


## Current Status

This work will be moving forward through different basic versions and there is little guarantee that things will remain the same from version to version. 
