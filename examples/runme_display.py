from similarity import tSNE, Jaccard, Distance
from transfer_learning import TransferLearning
from transfer_learning_display import TransferLearningDisplay

filename = 'test.pck'

tl = TransferLearning.load(filename)

# Calculate hte similarity
#similarity = Jaccard()
similarity = tSNE(display_type='hexbin')
#similarity = Distance(metric='cityblock')

# Create the display with the calculated similarity
tld = TransferLearningDisplay(similarity)

# Display the figure
tld.show(tl.fingerprints)
