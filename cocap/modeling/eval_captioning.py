# -*- coding: utf-8 -*-
# @Time    : 6/17/25
# @Author  : Yaojie Shen
# @Project : CoCap
# @File    : eval_captioning.py
import logging

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.spice.spice import Spice
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

logger = logging.getLogger(__name__)


class EvalCap:
    def __init__(self, annos, rests, cls_tokenizer=PTBTokenizer,
                 use_scorers=('Bleu', 'METEOR', 'ROUGE_L', 'CIDEr')):
        self.evalImgs = []
        self.eval = {}
        self.imgToEval = {}
        self.annos = annos
        self.rests = rests
        self.Tokenizer = cls_tokenizer
        self.use_scorers = use_scorers

    def evaluate(self):
        res = {}
        for r in self.rests:
            res[str(r['image_id'])] = [{'caption': r['caption']}]

        gts = {}
        for imgId in self.annos:
            gts[str(imgId)] = [{'caption': c} for c in self.annos[imgId]]

        # =================================================
        # Set up scorers
        # =================================================
        # print('tokenization...')
        tokenizer = self.Tokenizer()
        gts = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)
        # =================================================
        # Set up scorers
        # =================================================
        # print('setting up scorers...')
        use_scorers = self.use_scorers
        scorers = []
        if 'Bleu' in use_scorers:
            scorers.append((Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]))
        if 'METEOR' in use_scorers:
            scorers.append((Meteor(), "METEOR"))
        if 'ROUGE_L' in use_scorers:
            scorers.append((Rouge(), "ROUGE_L"))
        if 'CIDEr' in use_scorers:
            scorers.append((Cider(), "CIDEr"))
        if 'SPICE' in use_scorers:
            scorers.append((Spice(), "SPICE"))

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, gts.keys(), m)
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, gts.keys(), method)
        self.setEvalImgs()

    def setEval(self, score, method):
        self.eval[method] = score

    def setImgToEvalImgs(self, scores, imgIds, method):
        for imgId, score in zip(imgIds, scores):
            if not imgId in self.imgToEval:
                self.imgToEval[imgId] = {}
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId][method] = score

    def setEvalImgs(self):
        self.evalImgs = [eval for imgId, eval in self.imgToEval.items()]


def evaluate(submission, reference):
    """Score generated captions against the references.

    Only the videos present in *both* are scored. The scorers assert that prediction and
    reference keys match exactly, so a partial pass over the validation set - Lightning's
    sanity check, or `limit_val_batches` - would otherwise fail with a bare AssertionError
    rather than simply scoring what it has.
    """
    tokenizer = PTBTokenizer  # for English
    data = submission['results']

    rests = [{'image_id': str(name), 'caption': value[0]['sentence']}
             for name, value in data.items()]
    predicted = {r['image_id'] for r in rests}
    annos = {str(k): v for k, v in reference.items() if str(k) in predicted}

    unknown = predicted - set(annos)
    if unknown:
        # a prediction whose id is not in the references cannot be scored; if this fires for
        # more than a handful, the id conventions of the two sides have diverged
        logger.warning("%d predicted ids are absent from the references and will be skipped "
                       "(e.g. %s)", len(unknown), sorted(unknown)[:3])
        rests = [r for r in rests if r['image_id'] in annos]

    if not rests:
        logger.warning("nothing to score: no predicted id matched a reference")
        return {}
    if len(annos) < len(reference):
        logger.info("scoring %d of %d validation videos (partial pass)",
                    len(annos), len(reference))

    eval_cap = EvalCap(annos, rests, tokenizer)
    eval_cap.evaluate()

    return dict(eval_cap.eval)
