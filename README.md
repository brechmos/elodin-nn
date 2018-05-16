# Transfer Learning for Hubble Images

This code will read in a set of images, create cutouts, calculate "fingerprints" for each cutout and then do similarity data reduction.

The general idea is given a Hubble image (for example a thumbnail), create a cutout of it. Then this cutout image is passed through the Resnet50 calculator in order to determine the probability information based on ImageNet images.  Then, given all the fingerprints, use the tSNE data reduction method (or other method) in order to calculate the similarity between the fingerprints, and therefore the similarity between the cutouts.

![](https://github.com/brechmos-stsci/transfer-learning/raw/master/images/screenshot1.jpg)

![](https://github.com/brechmos-stsci/transfer-learning/raw/master/images/screenshot2.jpg)

## Simple example of the code

```
print('Setting up the data structure required')
gray_scale = DataGrayScale()
data = []
for fileinfo in np.random.choice(processing_dict, 300, replace=False):
    im = Data(location=fileinfo['location'], radec=fileinfo['radec'], meta=fileinfo['me
ta'])
    im.add_processing(gray_scale.save())
    data.append(im)
    db.save('data', im)

#
#  Create cutouts
#
print('Creating the Full image cutout generator')
full_cutout = FullImageCutoutGenerator(output_size=(224, 224))

cutout_crop = CutoutCrop([15, -15, 15, -15])
cutout_resize = CutoutResize([224, 224])

print('Going to create the cutouts')
cutouts = []
for datum in data:
    cutout = full_cutout.create_cutouts(datum)

    # Add the processing
    cutout.add_processing(cutout_crop)
    cutout.add_processing(cutout_resize)

    db.save('cutout', cutout)
    cutouts.append(cutout)

# Create the fingerprint calculator... fingerprint
print('Creating the info for the fingerprint calculator')
fresnet = FingerprintCalculatorResnet()
fc_save = fresnet.save()

print('Calculating the fingerprints')
fingerprints = fingerprint_calculate(cutouts, fc_save)
[db.save('fingerprint', x) for x in fingerprints]

print('Calculating the tSNE similarity')
similarity_tsne = similarity_calculate(fingerprints, 'tsne')
db.save('similarity', similarity_tsne)

```

## Current Status

This work will be moving forward through different basic versions and there is little guarantee that things will remain the same from version to version. 
