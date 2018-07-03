import numpy as np
import os
from recsysrw import shared
import regex
import re
from whoosh import index
from whoosh.fields import TEXT, STORED, Schema
from whoosh.analysis import SpaceSeparatedTokenizer, LowercaseFilter, StopFilter, StemFilter
from whoosh.qparser import QueryParser
from tqdm import trange
from collections import Counter

# Modified stoplist
stoplist = ['and', 'is', 'it', 'an', 'as', 'at', 'have', 'in', 'yet', 'if', 'when', 'by', 'to', 'be', 'we', 'a', 'on', 'of', 'will', 'can', 'the', 'or', 'are']


def normalize_title(name):
    name = name.lower()
    # Add spaces around every emoji
    name = regex.sub(r'(\p{So}\p{Sk}*)', r' \1 ', name)
    name = re.sub(r"[.,\/#!$%\^\*;:{}=\_`~()@']", ' ', name)
    name = re.sub(r'\s+', ' ', name).strip()
    # Truncate character repetitions (e.g. littt, happyyyy)
    name = re.sub(r'(.)\1{2,}', r'\1\1', name)
    return name


class TitleIndexer(object):
    def __init__(self, index_path, names=None):
        self._analyzer = SpaceSeparatedTokenizer() | LowercaseFilter() | StopFilter(minsize=1,
                                                                                    stoplist=stoplist) | StemFilter()
        if index.exists_in(index_path):
            self._ix = index.open_dir(index_path)
        else:
            self.build_index(index_path, names)
        self._qp = QueryParser("title", self._ix.schema, plugins=[])


    def build_index(self, path, names=None):
        print("Building index..")
        if not os.path.exists(path):
            os.makedirs(path)

        schema = Schema(title=TEXT(analyzer=self._analyzer), pid=STORED())
        titles = shared.graph.playlist_titles if names is None else names
        normalized_titles = [normalize_title(title) for title in titles]
        ix = index.create_in(path, schema)
        writer = ix.writer()
        for i in trange(len(normalized_titles)):
            title = normalized_titles[i]
            writer.add_document(title=title, pid=i)
        print("Committing..")
        writer.commit()
        print("Done.")
        self._ix = ix

    def title_to_tokens(self, title):
        return [t.text for t in self._analyzer(normalize_title(title))]

    def find_top_similar(self, title, threshold=5.5, return_scores=False):
        normalized_title = normalize_title(title)
        with self._ix.searcher() as searcher:
            query = self._qp.parse(normalized_title)
            results = searcher.search(query, limit=None)
            top_similar = []
            scores = []
            for r in results:
                if r.score < threshold:
                    break
                top_similar.append(r['pid'])
                scores.append(r.score)
            if return_scores:
                return np.asarray(top_similar), np.asarray(scores)
            else:
                return np.asarray(top_similar)


def expand_on_title(title, indexer: TitleIndexer, frequency_threshold=0.0, docsim_threshold=5, max_expand=100,
                    disallowed_tracks={}, disallowed_playlist=None):
    result_playlists = indexer.find_top_similar(title, threshold=docsim_threshold)
    track_counter = Counter()
    for result_pid in result_playlists:
        if result_pid == disallowed_playlist:
            continue        # no cheating
        for track in shared.graph.neighbors(result_pid):
            if track not in disallowed_tracks:
                track_counter[track] += 1
    frequencies = np.asarray([cnt for (_, cnt) in track_counter.most_common()], dtype=float) / len(result_playlists)
    expansion = np.asarray([track for (track, _) in track_counter.most_common()])
    expansion = expansion[np.where(frequencies >= frequency_threshold)]
    frequencies = frequencies[np.where(frequencies >= frequency_threshold)]
    if max_expand is not None:
        expansion = expansion[:max_expand]
        frequencies = frequencies[:max_expand]
    return expansion, frequencies


