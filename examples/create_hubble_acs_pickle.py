from astroquery.mast import Observations
import pickle
import os

# Basedir if we are on the public system
# basedir = '/ifs/operations/hst/public'
# filename = os.path.join(basedir, obsid[:4], obsid, obsid+'_drz_small.jpg')

# Basedir on my mac
basedir = '/Users/crjones/christmas/hubble/ACSimages/data'

obs = Observations.query_criteria(dataproduct_type=["image"], calib_level=3, 
        obs_collection='HST', instrument_name='ACS/WFC')

datadict = []
for o in obs[:20000]:
    if o['obs_id'][-1] == '0':

        # Create the filename
        obsid = o['obs_id'].lower()
        filename = os.path.join(basedir, obsid+'_drz_small.jpg')

        # Only add if the file exists
        if os.path.exists(filename):
            datadict.append(
                {
                    'filename': filename,
                    'radec': (o['s_ra'], o['s_dec']),
                    'meta': {cn: o[cn] for cn in obs.colnames}
                }
            )

# Write it out.
print('Writing out {} data elements'.format(len(datadict)))
pickle.dump(datadict, open('hubble_acs.pck', 'wb'))
