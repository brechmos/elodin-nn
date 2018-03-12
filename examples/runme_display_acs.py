from similarity import tSNE, Jaccard, Distance
from transfer_learning import TransferLearning
from transfer_learning_display import TransferLearningDisplay
import pickle

# Load the Pickle file of data results
#filename = 'acs_19152.pck'
filename = 'acs_10000.pck'
tl = TransferLearning.load(filename)

# Load the dictionary of RA/DEC information
radec = pickle.load(open('/Users/crjones/christmas/hubble/ACSimages/data/radec.pck', 'rb'))

# Calculate hte similarity
#similarity = Jaccard()
similarity = tSNE(display_type='hexbin')
#similarity = tSNE()
#similarity = Distance(metric='cityblock')

# Create the display with the calculated similarity
tld = TransferLearningDisplay(similarity, ra_dec=radec)

# Display the figure
tld.show(tl.fingerprints)
