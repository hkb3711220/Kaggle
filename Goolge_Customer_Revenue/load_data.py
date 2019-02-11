import os
import json
import pandas as pd
from tqdm import tqdm

os.chdir(os.path.dirname(__file__))

class load_data(object):
    """

    fullVisitorId- A unique identifier for each user of the Google Merchandise Store.
    channelGrouping - The channel via which the user came to the Store.
    date - The date on which the user visited the Store.
    device - The specifications for the device used to access the Store.
    geoNetwork - This section contains information about the geography of the user.
    socialEngagementType - Engagement type, either "Socially Engaged" or "Not Socially Engaged".
    totals - This section contains aggregate values across the session.
    trafficSource - This section contains information about the Traffic Source from which the session originated.
    visitId - An identifier for this session. This is part of the value usually stored as the _utmb cookie. This is only unique to the user. For a completely unique ID, you should use a combination of fullVisitorId and visitId.
    visitNumber - The session number for this user. If this is the first session, then this is set to 1.
    visitStartTime - The timestamp (expressed as POSIX time).
    hits - This row and nested fields are populated for any and all types of hits. Provides a record of all page visits.
    customDimensions - This section contains any user-level or session-level custom dimensions that are set for a session. This is a repeated field and has an entry for each dimension that is set.
    totals - This set of columns mostly includes high-level aggregate data.

    """

    def __init__(self, target, rebuild=True):

        self.datafile = target + '_v2.csv'
        self.path = './csv'
        self.rebuild = rebuild

        if os.path.exists(self.path) == False:
            os.mkdir(self.path)

    def _read_csv(self, col_name, chunksize):

        save_path = os.path.join(self.path, col_name + '.csv')

        if os.path.exists(save_path) == False or self.rebuild == True:
            output = pd.DataFrame()
            for df in tqdm(pd.read_csv(self.datafile, chunksize=chunksize)):
                tar = df[col_name].values
                #print(tar)
                for v in tar:
                    v = json.loads(v) # Change String to dictionary
                    v = pd.DataFrame.from_dict(v, orient='index')
                    v = v.T #transpose Dataframe
                    output = output.append(v, ignore_index=True)
                break

            output.to_csv(save_path, index=False)
            return output

        else:
            output = pd.read_csv(save_path)
            return output

#total = load_data(target='train', rebuild=False)._read_csv('totals', chunksize=600)
