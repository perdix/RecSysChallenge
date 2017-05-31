mongoexport.exe --db spotify_data --collection artist_data --out "artists_metadata.csv" --type csv --fields id,genres.0
