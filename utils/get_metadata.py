import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import os
import time
import json
from tqdm import trange
import argparse
import collections

client_id = os.environ['SPOTIPY_CLIENT_ID']
client_secret = os.environ['SPOTIPY_CLIENT_SECRET']
SPOTIPY_REDIRECT_URI = 'http://localhost:8080'
uri_histogram = collections.Counter()

audio_features = {'acousticness',
                  'danceability',
                  'energy',
                  'instrumentalness',
                  'liveness',
                  'speechiness',
                  'tempo',
                  'valence'}


def process_mpd(path, quick=False, max_files_for_quick_processing=1):
    start = time.time()
    count = 0
    filenames = os.listdir(path)
    tracks_albums_artists = set()
    for filename in sorted(filenames):
        if filename.startswith("mpd.slice.") and filename.endswith(".json"):
            print("Processing ", filename)
            fullpath = os.sep.join((path, filename))
            f = open(fullpath)
            js = f.read()
            f.close()
            mpd_slice = json.loads(js)
            for playlist in mpd_slice['playlists']:
                for track in playlist['tracks']:
                    tracks_albums_artists.add((track['track_uri'], track['album_uri'], track['artist_uri']))
                    uri_histogram[track['track_uri']] += 1
            count += 1

            if quick and count >= max_files_for_quick_processing:
                break
    print("Time: ", time.time() - start)
    return tracks_albums_artists


def get_metadata(track_uris):
    metadata = {}
    start = time.time()
    for i in trange(0, len(track_uris), 50):
        batch_tracks = track_uris[i:i + 50]
        audio_feats = sp.audio_features(tracks=batch_tracks)
        tracks_meta = sp.tracks(tracks=batch_tracks)['tracks']
        for j, (features, meta) in enumerate(zip(audio_feats, tracks_meta)):
            track_info = {}
            if meta is not None:
                if 'explicit' in meta:
                    track_info['explicit'] = meta['explicit']
                if 'name' in meta:
                    track_info['name'] = meta['name']
                if 'popularity' in meta:
                    track_info['popularity'] = meta['popularity']
                if 'album' in meta and 'release_date' in meta['album']:
                    track_info['release_date'] = meta['album']['release_date']

            if features is not None:
                for key in audio_features:
                    if key in features:
                        track_info[key] = features[key]

            track_info['count'] = uri_histogram[batch_tracks[j]]

            if batch_tracks[j] == 'spotify:track:0UaMYEvWZi0ZqiDOoHU3YI':
                print(track_info)

            metadata[batch_tracks[j]] = track_info

    end = time.time()
    print("Took %i s" % (end - start))
    return metadata


def main(mpd_path, target_file, quick=False):
    tracks_albums_artists = process_mpd(mpd_path, quick)
    track_uris, album_uris, artist_uris = zip(*tracks_albums_artists)
    print('spotify:track:0UaMYEvWZi0ZqiDOoHU3YI' in track_uris)
    metadata = get_metadata(track_uris)
    print('spotify:track:0UaMYEvWZi0ZqiDOoHU3YI' in metadata)

    with open(target_file, 'w') as f:
        json.dump(metadata, f, indent=4, separators=(',', ': '))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Retrieve the track metadata from Spotify for all tracks in the MPD.')
    parser.add_argument('mpd_path', type=str, help="Path to the MPD")
    parser.add_argument('target_file', type=str, help="Target filename where metadata will be written.")
    parser.add_argument('-quick', action='store_true', default=False, help='Limits the number of MPD files '
                                                                                       'read for quick testing.')

    args = parser.parse_args()

    client_credentials_manager = SpotifyClientCredentials()
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
    sp.trace = False

    main(args.mpd_path, args.target_file, args.quick)
