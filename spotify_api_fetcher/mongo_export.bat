mongoexport.exe --db spotify_data --collection track_data --out "tracks_metadata.csv" --type csv --fields id,name,artists.0.name,album.name,album.album_type,track_number,explicit,popularity
