import time

def get_range_for_rank(n, word_size, rank):
    base, reminder = n // word_size, n % word_size
    start, end = base * rank, base * (rank + 1)
    if rank == word_size - 1:
        end += reminder
    return start, end

class GetBestSegmentsCombinations:
    def __init__(self, tokendb, min_n_token, max_n_token, max_combinations_per_iter=2, max_time_consumption=1, forbid_self=False):
        self.tokendb = tokendb
        self.min_n_token = min_n_token
        self.max_n_token = max_n_token
        self.max_combinations_per_iter = max_combinations_per_iter
        self.max_time_consumption = max_time_consumption
        self.forbid_self = forbid_self
    
    def __get_best_segments_combinations(self, start, end, max_n_token, uttid=None):

        tokens_ = self.tokens[start:end]
        if len(tokens_) == 0:
            return [[]]
        
        # invalid segments are stored to reduce repeat computation
        # segments combinations are not stored to limit memory consumption
        if len(tokens_) < self.min_n_token or ((start, end) in self.segment2combinations and self.segment2combinations[(start, end)] is None):
            return []
        
        if (start, end) in self.segment2combinations:
            return self.segment2combinations[(start, end)]

        segments_combinations = []
        for n_token in range(max_n_token, self.min_n_token - 1, -1):
            if self.should_stop_iter:
                break
            for pos in range(0, len(tokens_) - n_token + 1):
                if self.should_stop_iter or len(segments_combinations) >= self.max_combinations_per_iter:
                    break
                token = tokens_[pos:pos+n_token]

                condition = token in self.tokendb[n_token]
                if self.forbid_self:
                    import pdb; pdb.set_trace()
                    condition = condition and any((segment[0] != uttid for segment in self.tokendb[n_token][token]))

                if condition:
                    prev_segments_combinations = self.__get_best_segments_combinations(start, start + pos, n_token - 1)
                    if len(prev_segments_combinations) == 0:
                        continue
                    next_segments_combinations = self.__get_best_segments_combinations(start + pos + n_token, end, max_n_token)
                    if len(next_segments_combinations) == 0:
                        continue

                    for prev_segment in prev_segments_combinations:
                        for next_segment in next_segments_combinations:
                            segments_combinations.append(prev_segment + [token] + next_segment)
            if len(segments_combinations) != 0:
                break                

        if len(segments_combinations) == 0:
            self.segment2combinations[(start, end)] = None
        else:
            self.segment2combinations[(start, end)] = segments_combinations

        return segments_combinations        
        
    @property
    def should_stop_iter(self):
        return time.time() - self.start > self.max_time_consumption

    def __call__(self, tokens, uttid=None):
        self.tokens = tokens
        self.segment2combinations = dict()
        
        self.start = time.time()
        best_segments_combinations = self.__get_best_segments_combinations(0, len(tokens), self.max_n_token, uttid)
        if len(best_segments_combinations) == 0:
            return []

        best_segments_length = min([len(segments) for segments in best_segments_combinations])
        best_segments_combinations = [segments for segments in best_segments_combinations if len(segments) == best_segments_length]
        return best_segments_combinations


class GenerateAudio:
    def __init__(self, uttid2audio, token2segments, uttid2combinations, load_audio=None):
        self.uttid2audio = uttid2audio
        self.token2segments = token2segments
        self.load_audio = kaldiio.load_mat if load_audio is None else load_audio
        self.uttid2combinations = uttid2combinations

    def __call__(self, uttid):

        # pdb.set_trace()
        synthesis_segments = []
        if uttid not in self.uttid2combinations:
            raise ValueError(f"combination for uttid {uttid} has not been computed")

        segment_combination = random.choice(self.uttid2combinations[uttid])
        for token in segment_combination:
            segment = random.choice(self.token2segments[token])
            uttid, start, end = segment
            rate, wave = self.load_audio(self.uttid2audio[uttid])
            synthesis_segment = wave[int(start * rate) : int(end * rate)]
            synthesis_segment = synthesis_segment / synthesis_segment.max() * 0.9
            synthesis_segments.append(synthesis_segment)

        synthesis_wave = np.concatenate(synthesis_segments)
        return synthesis_wave

