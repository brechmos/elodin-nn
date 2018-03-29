from astroquery.mast import Observations
import ujson
import os

basedir = '/ifs/operations/hst/public'

obs = Observations.query_criteria(dataproduct_type=["image"], calib_level=3, 
        obs_collection='HST', instrument_name='ACS/WFC')

datadict = {}
for o in obs[:20000]:
    if o['obs_id'][-1] == '0':

        # Create the filename
        obsid = o['obs_id'].lower()
        filename = os.path.join(basedir, obsid[:4], obsid, obsid+'_drz_small.jpg')

        # Only add if the file exists
        if True or os.path.exists(filename):
            datadict[obsid] = {
                'filename': filename,
                'radec': (o['s_ra'], o['s_dec'])#,
                #'meta': {cn: o[cn] for cn in obs.colnames}
            }

# Write it out.
ujson.dump(datadict, open('hubble_acs.json', 'wt'))
