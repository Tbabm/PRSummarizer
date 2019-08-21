import unittest
from rouge import Rouge
from rouge import rouge_score

class TestRougeScore(unittest.TestCase):
    def test_get_scores(self):
        rouge = Rouge()
        dec = ["""updates libraries and tools to latest versions and tools to latest versions . 
updates libraries and tools to latest versions . 
updates libraries and tools to latest versions . 
updates libraries and tools to latest versions . 
updates libraries and tools to latest versions . 
updates libraries and tools to latest versions . 
updates libraries and tools to latest versions . 
updates libraries and tools to latest versions . 
updates libraries and tools to latest versions . 
updates libraries and tools to latest versions . 
updates libraries and tools to latest versions . 
updates libraries and tools to latest versions""",
               """removed option to unmute other users removed option to unmute other users"""]
        ref = ["""just made a small adjustement on lint to allow it to pass .""",
               """the option to unmute other users is removed , while still leaving the functionality to mute yourself ."""]

        scores = rouge.get_scores(dec, ref)
        results = [{
            "rouge-1": {
                'r': 0.16667,
                'p': 0.02247,
                'f': 0.03960
            },
            "rouge-2": {
                'r': 0.0,
                'p': 0.0,
                'f': 0.0
            },
            "rouge-l": {
                "r": 0.16667,
                "p": 0.02247,
                "f": 0.03960
            }
        },
            {
            'rouge-1': {
                'r': 0.43750,
                'p': 0.58333,
                'f': 0.50000
            },
            'rouge-2': {
                'r': 0.26667,
                'p': 0.36364,
                'f': 0.30769
            },
            'rouge-l': {
                'r': 0.43750,
                'p': 0.58333,
                'f': 0.50000
            }
        }]
        print(scores)
        for score, result in zip(scores, results):
            for key1 in result:
                for key2 in result[key1]:
                    self.assertAlmostEqual(round(score[key1][key2],5), result[key1][key2])

    def test_split_sentence(self):
        text = "unmute other users &lt;this&gt; that - hehe hahah.xixi heihei/gg how_about 0 bike\'s !\"#$%&\'()*+, - ./:;<=>?@[\\]^_`{|}~"
        tokens = rouge_score._split_sentence(text)
        result = 'unmute other users lt this gt that - hehe hahah xixi heihei gg how about 0 bike s -'.split()
        for t, r in zip(tokens, result):
            self.assertEqual(t, r)


if __name__ == '__main__':
    unittest.main()
