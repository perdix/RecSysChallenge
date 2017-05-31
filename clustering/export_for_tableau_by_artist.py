import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import euclidean_distances
from sklearn.cluster import KMeans, DBSCAN

from sklearn.manifold import MDS
from sklearn.decomposition import PCA

def get_artist_vector(artist_name):
    tracks_by_artist = artists[artists['artist'] == artist_name]
    if tracks_by_artist.empty:
        return pd.DataFrame()

    return tracks_by_artist.groupby('artist').mean()

selected_artists = ["ABBY","AKI","Abroo","Ace Ventura & Lish","Adana Twins","Airhead","Alan Watts","Alex Smoke","Ali Khan","All Good Funk Alliance","Alle Farben","Ametsub","Anchorsong","Anders Trentemoller","Andras Fox","Angus & Julia Stone","Animaltrainer","Anndreyah Vargas","Aphex Twin","Ark Patrol","Arkist","Axel Boman","Ayla Nereo","Baaz","Bag Raiders","Barry Louis Polisar","Bastille","Beatenberg","Benjie","Blackbird Blackbird","Bluetech","Blumentopf","Bob Dylan","Bob Marley & The Wailers","Bobby McFerrin","Bon Iver","Bonobo","Booka Shade","Boris Dlugosch","Bostro Pesopeo","Brian Witzig","Buffalo Springfield","Bugge Wesseltoft","Bvdub","Cajuu","Calibre","Capital Cities","Cari Golden","Caribou","Causes","Champion","Chelonis R. Jones","Chet Faker","Chopstick","Chopstick & Johnjon","Christine Hoberg","Cinnamon Chasers","Claptone","Claude VonStroke","Clean Bandit","Coleman Hell","Colours in the Street","Commix","Cosmo Sheldrake","Croatia Squad","Croquet Club","Crowded House","Crystal Fighters","D Dub","DJ Cam Quartet","DJ Clock","DJ Crazy J Rodriguez","Dabrye","Daft Punk","Daniel Bortz","Daniel Dexter","Daniel Mehlhart","Daniel Steinberg","Daniel Symons","Dark Sky","Dauwd","David August","David Guetta","De Hofnar","Death On The Balcony","Debruit","Deniz Kurtel","Dirty Doering","Diversion","Dolle Jolle","Dousk","Dua Lipa","Eastern Sun","Ebo Taylor","Eboi","Edamame","Efdemin","Ejeca","Eliot Lipp","Eliot Short","Emancipator","Embody","Fat Freddy's Drop","Feist","Felix Jaehn","Few Nolder","Filtertypen","Finnebassen","First Aid Kit","Flight Facilities","Flomega","Flooting Grooves","Florian Meindl","Flume","Foster The People","Four Tet","Francesco Leali","Frank Wiedemann","Fritz Helder","Fritz Kalkbrenner","Frivolous","Fuckpony","Fugees","Funk","Funny Van Dannen","Future Disco","Future Prophecies","GTA","Gallant","Gelka","General Midi","Gentleman","George Maple","Gidge","Gonzales","Gramatik","Grandbrothers","Gregor Tresher","Groundislava","Grouplove","Guts","Harold van Lennep","Hauschka","Haze","Henrik Schwarz","Herbert Pixner Projekt","Hird","Hot Chip","Icarus","Ice Cube","Iglu & Hartly","Isolée","Israel Kamakawiwo'ole","Jafu","Jakatta","James Arthur","James Brown & The Famous Flames","James Flavour + ESB","James Hersey","Jamie Prado","Jamie xx","Jasmine Thompson","Jess Glynne","Jezzabell Doran","Jimpster","Joachim Pastor","Job 2 do","John Legend","John Lennon","Johnjon","Joris Delacroix","Josef Salvat","Josh Martinez","José González","Juliet Sikora","Julio Bashmore","Junge Junge","Jungle","Kaiori Breathe","Kalabrese","Kalipo","Kapten Röd","Kasper Bjørke","Kellerkind","Ken Hayakawa feat. Herb Piper","Kenton Slash Demon","Kevin Over","Kiasmos","Kidkanevil","Kiki","Kim Brown","Kimyan Law","Kirtaniyas","Kollektiv Ost","Kollektiv Turmstrasse","Koreless","Kyle Pearce","LA Priest","Lack Of Afro","Lafa Taylor","Laid Back","Lakechild","Larissa Kapp","Lee \"Scratch\" Perry","Lemonchill","Lions Head","Little People","Lizzy Plotkin","London Grammar","Lou Reed","Luc Angenehm","Ludovico Einaudi","Lynx","M83","MC Hammer","MGMT","Madelyn Grant","Maduk","Mano Le Tough","Manuel Tur","Marcapasos","Marcus Layton","Marek Hemmann","Marie Beeckman","Mario","Marsha Ambrosius","Martin Jondo","Martin Solveig","Matthias Vogt","Max Kunz","Maya Jane Coles","Megaloh","Meggy","Metaform","MiM0SA","Michael Jackson","Michael Schulte","Moar","Moderat","Mokhov","Monkey Safari","Monkeybrain","Moon Holiday","Mount Kimbie","Mr. Green","Mr. Lif","Mr.Troy","Mutt","NYM","Neon Indian","Nhan Solo","Nicolas Haelg","Nicolette Krebitz","Niklas Ibach","Nils Penner","Nina Simone","Niva","Noir","Nutia","Nôze","ODESZA","OVERWERK","Oliver Schories","Orbion","Ost & Kjex","OutKast","Ovasoul7","Oxia","Panda Bear","Pantha Du Prince","Paper Tiger","Paskal & Urban Absolutes","Pat Thomas","Patrick Chardronnet","Peer Kusiv","Peia","Perttu","Peter Horn Jr.","Peter Tosh","Phaeleh","Pharrell Williams","Phil Dennys","Phon.o","PillowTalk","Pimf","Prestige Collection","Pretty Lights","Qtier","Raashan Ahmad","Radio Diffusion","Rampue","Random Rab","Ratatat","Reason The Citizen","Retro Stefson","Rico Puestel","Robag Wruhme","Robert Babicz","Robot Koch","Rodriguez Jr.","Rodriguez Jr.","Rodríguez","Ryan Herr","SG Lewis","SOHN","Sam Cooke","Sam Martin","Sam Sure","Saqi","Sascha Funke","Saschienne","Sean Haefeli","Sekuoia","Severin Borer","Sheppard","Siedah Garrett","Siriswad","Slugabed","Smash TV","Snoop Dogg","Solomun","SoulChef","Stanton Warriors","Steve Miller Band","Steve Winwood","Stimming & David August","Sting","Supervision","Synapson","Synkro","T-Shirt","TM Juke","Ta-ku","Talking Heads","Tantsui","Tara Brooks","Taylor Mcferrin","Teaspoon","Tender Games","Terranova","The Avener & Phoebe Killdeer","The Box Tops","The Chemical Brothers","The Jacksons","The Kingpins","The Lumineers","The Polish Ambassador","The Wailers","The Waves","The Weeknd","The Whitest Boy Alive","The Wood Brothers","The xx","Thieves Like Us","Thugfucker","Till Von Sein","Tim Hanmann","Timid Tiger","Tin Sparrow","Tipper","Tommy Sparks","Tony Humphries","Tor","Torsten Bo Jacobsen","Tosca","Tourist","Tracy Chapman","Train","Tube & Berger","Tuff City Kids","Turtle","Twenty One Pilots","Tycho","Udo Kier","Umse","Underset","Urban Cone","VIMES","Vadoinmessico","Van Morrison","Viken Arman","Voyeur","Weval","Will Cady","Will Smith","Willow","Wolf + Lamb","Yagya","Yann Tiersen","Yarah Bravo","Yoga Soul","Yukimi Nagano","Yusuf / Cat Stevens","ZHU","Zaz","Zonderling","ambinate","and.id","Ásgeir"]

tracks_with_names = pd.read_csv('../data/tracks_with_metadata.csv', encoding='utf-8')
tracks_with_names = tracks_with_names[(tracks_with_names['duration'] > 120)]

columns = ['energy', 'liveness', 'speechiness', 'acousticness', 'danceability', 'valence', 'instrumentalness']


# group by artist, take groups with > x (eg. 10) tracks, then take a sample of y (eg. 100) artists
#artists = tracks_with_names.groupby('artist').filter(lambda x: len(x) > 20 and len(x) < 50).groupby('artist').mean().sample(1)
track_count_by_artists = tracks_with_names.groupby('artist').size().to_frame('track_count')
artists = tracks_with_names.groupby('artist').mean().join(track_count_by_artists).reset_index()

artists_to_export = artists.sample(1).set_index('artist')

for artist_name in selected_artists:
    artists_to_export = artists_to_export.append(get_artist_vector(artist_name))

# artists = artists.append(tracks_with_names[tracks_with_names['artist'] == '2Pac'].iloc[0:25].groupby('artist').mean().assign(genre='Hip-Hop'))
# artists = artists.append(tracks_with_names[tracks_with_names['artist'] == 'Jay-Z'].iloc[0:25].groupby('artist').mean().assign(genre='Hip-Hop'))
# artists = artists.append(tracks_with_names[tracks_with_names['artist'] == 'Dr. Dre'].iloc[0:25].groupby('artist').mean().assign(genre='Hip-Hop'))
# artists = artists.append(tracks_with_names[tracks_with_names['artist'] == '50 Cent'].iloc[0:25].groupby('artist').mean().assign(genre='Hip-Hop'))
# artists = artists.append(tracks_with_names[tracks_with_names['artist'] == 'D12'].iloc[0:25].groupby('artist').mean().assign(genre='Hip-Hop'))
# artists = artists.append(tracks_with_names[tracks_with_names['artist'] == 'DMX'].iloc[0:25].groupby('artist').mean().assign(genre='Hip-Hop'))
# artists = artists.append(tracks_with_names[tracks_with_names['artist'] == 'Xzibit'].iloc[0:25].groupby('artist').mean().assign(genre='Hip-Hop'))
# artists = artists.append(tracks_with_names[tracks_with_names['artist'] == 'Snoop Dogg'].iloc[0:25].groupby('artist').mean().assign(genre='Hip-Hop'))
#
#
# artists = artists.append(tracks_with_names[tracks_with_names['artist'] == 'Stimming'].iloc[0:25].groupby('artist').mean().assign(genre='Electronic Music'))
# artists = artists.append(tracks_with_names[tracks_with_names['artist'] == 'Kollektiv Turmstrasse'].iloc[0:25].groupby('artist').mean().assign(genre='Electronic Music'))
# artists = artists.append(tracks_with_names[tracks_with_names['artist'] == 'Rodriguez Jr.'].iloc[0:25].groupby('artist').mean().assign(genre='Electronic Music'))
# artists = artists.append(tracks_with_names[tracks_with_names['artist'] == 'David August'].iloc[0:25].groupby('artist').mean().assign(genre='Electronic Music'))
# artists = artists.append(tracks_with_names[tracks_with_names['artist'] == 'Solomun'].iloc[0:25].groupby('artist').mean().assign(genre='Electronic Music'))
#
#
# artists = artists.append(tracks_with_names[tracks_with_names['artist'] == 'Britney Spears'].iloc[0:25].groupby('artist').mean().assign(genre='Pop'))
# artists = artists.append(tracks_with_names[tracks_with_names['artist'] == 'Pink'].iloc[0:25].groupby('artist').mean().assign(genre='Pop'))
# artists = artists.append(tracks_with_names[tracks_with_names['artist'] == 'Christina Aguilera'].iloc[0:25].groupby('artist').mean().assign(genre='Pop'))
# artists = artists.append(tracks_with_names[tracks_with_names['artist'] == 'Katy Perry'].iloc[0:25].groupby('artist').mean().assign(genre='Pop'))
# artists = artists.append(tracks_with_names[tracks_with_names['artist'] == 'Gwen Stefani'].iloc[0:25].groupby('artist').mean().assign(genre='Pop'))


df_features = artists_to_export[columns]

# dimensionality reduction with MDS
mds = MDS(n_components=2, dissimilarity='precomputed')

similarities_eculidean = euclidean_distances(df_features)
reduced_data_eculidean = mds.fit(similarities_eculidean, df_features).embedding_

artists_to_export['reduced_x_mds'] = reduced_data_eculidean[:, 0]
artists_to_export['reduced_y_mds'] = reduced_data_eculidean[:, 1]


# dimensionality reduction with PCA
pca = PCA(n_components=2)
reduced_data_pca = pca.fit_transform(df_features)

artists_to_export['reduced_x_pca'] = reduced_data_pca[:, 0]
artists_to_export['reduced_y_pca'] = reduced_data_pca[:, 1]


# clustering with kmeans
kmeans = KMeans(n_clusters=8, random_state=0)
kmeans.fit(df_features)

artists_to_export['cluster_kmeans'] = pd.Series(kmeans.labels_, index=artists_to_export.index)


artists_to_export.to_csv('for_tableau_by_artist.csv', encoding='utf-8')

print('theend')

