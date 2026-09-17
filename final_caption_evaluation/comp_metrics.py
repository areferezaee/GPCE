import re
from pathlib import Path

import numpy as np
import pandas as pd

from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = "captions_examples.txt"

SUMMARY_CSV = "caption_metrics_summary.csv"
PER_SAMPLE_CSV = "caption_metrics_per_sample.csv"
CLINICAL_CSV = "clinical_metrics_detailed.csv"


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text: str) -> str:
    if text is None:
        return ""

    text = text.lower()
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    text = re.sub(r"#{3,}", " ", text)
    text = re.sub(r"`+", " ", text)
    text = re.sub(r"\s+", " ", text)

    text = re.sub(
        r"[^\w\s\.,;:\-\(\)%]",
        " ",
        text
    )

    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize(text: str):
    return normalize_text(text).split()


# ============================================================
# TXT PARSER
# ============================================================

def parse_caption_file(file_path: str):

    text = Path(file_path).read_text(
        encoding="utf-8"
    )

    text = text.replace(
        "\r\n",
        "\n"
    ).replace(
        "\r",
        "\n"
    )

    sample_pattern = re.compile(
        r"(?m)^(0X[0-9A-Fa-f]+):\s*$"
    )

    matches = list(
        sample_pattern.finditer(text)
    )

    samples = []

    for i, match in enumerate(matches):

        sample_id = match.group(1)

        start = match.end()

        end = (
            matches[i + 1].start()
            if i + 1 < len(matches)
            else len(text)
        )

        block = text[start:end].strip()

        gt_match = re.search(
            r"(?ms)^\s*GT:\s*(.*?)(?=^\s*before:\s*)",
            block
        )

        before_match = re.search(
            r"(?ms)^\s*before:\s*(.*?)(?=^\s*after:\s*)",
            block
        )

        after_match = re.search(
            r"(?ms)^\s*after:\s*(.*)$",
            block
        )

        gt = (
            gt_match.group(1).strip()
            if gt_match
            else ""
        )

        before = (
            before_match.group(1).strip()
            if before_match
            else ""
        )

        after = (
            after_match.group(1).strip()
            if after_match
            else ""
        )

        samples.append({
            "id": sample_id,
            "gt": gt,
            "before": before,
            "after": after,
        })

    return samples


# ============================================================
# BLEU
# ============================================================

def calculate_bleu(
    references,
    predictions
):

    smooth = SmoothingFunction().method1

    refs = [
        [tokenize(ref)]
        for ref in references
    ]

    hyps = [
        tokenize(pred)
        for pred in predictions
    ]

    weights = {

        "BLEU-1": (
            1.0,
            0.0,
            0.0,
            0.0
        ),

        "BLEU-2": (
            0.5,
            0.5,
            0.0,
            0.0
        ),

        "BLEU-3": (
            1 / 3,
            1 / 3,
            1 / 3,
            0.0
        ),

        "BLEU-4": (
            0.25,
            0.25,
            0.25,
            0.25
        ),
    }

    results = {}

    for name, weight in weights.items():

        score = corpus_bleu(
            refs,
            hyps,
            weights=weight,
            smoothing_function=smooth
        )

        results[name] = float(score)

    return results


# ============================================================
# ROUGE-L
# ============================================================

def calculate_rouge_l(
    references,
    predictions
):

    scorer = rouge_scorer.RougeScorer(
        ["rougeL"],
        use_stemmer=True
    )

    scores = []

    for ref, pred in zip(
        references,
        predictions
    ):

        result = scorer.score(
            normalize_text(ref),
            normalize_text(pred)
        )

        scores.append(
            result["rougeL"].fmeasure
        )

    return float(
        np.mean(scores)
    )


# ============================================================
# METEOR
# ============================================================

def calculate_meteor(
    references,
    predictions
):

    scores = []

    for ref, pred in zip(
        references,
        predictions
    ):

        ref_tokens = tokenize(ref)
        pred_tokens = tokenize(pred)

        if not pred_tokens:

            scores.append(0.0)
            continue

        score = meteor_score(
            [ref_tokens],
            pred_tokens
        )

        scores.append(
            float(score)
        )

    return float(
        np.mean(scores)
    )


# ============================================================
# CIDEr
# ============================================================

def calculate_cider(
    references,
    predictions
):

    try:

        from pycocoevalcap.cider.cider import Cider

    except ImportError:

        print(
            "\n[WARNING] pycocoevalcap is not installed."
        )

        print(
            "CIDEr will be reported as NaN."
        )

        print(
            "Install with:"
        )

        print(
            "pip install pycocoevalcap\n"
        )

        return np.nan

    gts = {
        i: [
            normalize_text(ref)
        ]
        for i, ref in enumerate(references)
    }

    res = {
        i: [
            normalize_text(pred)
        ]
        for i, pred in enumerate(predictions)
    }

    cider = Cider()

    score, _ = cider.compute_score(
        gts,
        res
    )

    return float(score)


# ============================================================
# CONTENT-ORIENTED ONTOLOGY
#
# This ontology follows the actual fixed-format GT template.
# ============================================================

FIELD_CONFIG = {

    # --------------------------------------------------------
    # 1. CARDIAC STRUCTURE
    #
    # GT explicitly describes:
    # "heart"
    # "left ventricle"
    # --------------------------------------------------------

    "cardiac_structure": {

        "presence": [
            r"\bheart\b",
            r"\bleft ventricle\b",
            r"\bleft ventricular\b",
            r"\bventricle\b",
        ],

        "values": {

            "left_ventricle": [
                r"\bleft ventricle\b",
                r"\bleft ventricular\b",
            ],

            "heart": [
                r"\bheart\b",
            ],
        },
    },


    # --------------------------------------------------------
    # 2. DIASTOLIC PHASE
    # --------------------------------------------------------

    "diastolic_phase": {

        "presence": [
            r"\bdiastole\b",
            r"\bdiastolic\b",
        ],

        "values": {
            "diastole": [
                r"\bdiastole\b",
                r"\bdiastolic\b",
            ],
        },
    },


    # --------------------------------------------------------
    # 3. SYSTOLIC PHASE
    # --------------------------------------------------------

    "systolic_phase": {

        "presence": [
            r"\bsystole\b",
            r"\bsystolic\b",
        ],

        "values": {
            "systole": [
                r"\bsystole\b",
                r"\bsystolic\b",
            ],
        },
    },


    # --------------------------------------------------------
    # 4. DIASTOLIC FILLING
    # --------------------------------------------------------

    "diastolic_filling": {

        "presence": [
            r"\brelatively limited filling\b",
            r"\blimited filling\b",
            r"\bmoderate filling\b",
            r"\bsubstantial filling\b",
            r"\bfilling\b",
        ],

        "values": {

            # Most specific patterns first
            "relatively_limited": [
                r"\brelatively limited filling\b",
            ],

            "substantial": [
                r"\bsubstantial filling\b",
            ],

            "moderate": [
                r"\bmoderate filling\b",
            ],

            "generic_filling": [
                r"\bfilling\b",
                r"\bfilled\b",
                r"\bfills\b",
            ],
        },
    },


    # --------------------------------------------------------
    # 5. SYSTOLIC CAVITY REDUCTION
    # --------------------------------------------------------

    "systolic_cavity_reduction": {

        "presence": [
            r"\bmarked reduction in ventricular cavity size\b",
            r"\bmoderate reduction in ventricular cavity size\b",
            r"\breduction in ventricular cavity size\b",
            r"\breduced ventricular cavity size\b",
            r"\bdecreased ventricular cavity size\b",
            r"\bcavity size\b",
        ],

        "values": {

            "marked": [
                r"\bmarked reduction\b",
                r"\bmarkedly reduced\b",
            ],

            "moderate": [
                r"\bmoderate reduction\b",
                r"\bmoderately reduced\b",
            ],

            "reduced": [
                r"\breduced ventricular cavity\b",
                r"\breduced\b.*\bcavity size\b",
            ],

            "decreased": [
                r"\bdecreased ventricular cavity\b",
                r"\bdecreased\b.*\bcavity size\b",
            ],

            "generic_reduction": [
                r"\breduction\b",
                r"\breduced\b",
                r"\bdecreased\b",
            ],
        },
    },


    # --------------------------------------------------------
    # 6. POST-CONTRACTION CAVITY
    # --------------------------------------------------------

    "post_contraction_cavity": {

        "presence": [
            r"\bcavity becomes\b",
            r"\bafter contraction\b",
            r"\bpost[- ]contraction\b",
        ],

        "values": {

            "small": [
                r"\bcavity becomes small\b",
                r"\bsmall after contraction\b",
            ],

            "moderate": [
                r"\bcavity becomes moderate\b",
                r"\bmoderate after contraction\b",
            ],

            "substantial": [
                r"\bcavity becomes substantial\b",
                r"\bsubstantial after contraction\b",
            ],

            "generic_after_contraction": [
                r"\bafter contraction\b",
                r"\bcavity becomes\b",
            ],
        },
    },


    # --------------------------------------------------------
    # 7. VENTRICULAR EMPTYING
    # --------------------------------------------------------

    "ventricular_emptying": {

        "presence": [
            r"\bmoderately effective ventricular emptying\b",
            r"\beffective ventricular emptying\b",
            r"\blimited ventricular emptying\b",
            r"\bventricular emptying\b",
            r"\bemptying\b",
        ],

        "values": {

            "effective": [
                r"\beffective ventricular emptying\b",
                r"\beffective emptying\b",
            ],

            "moderately_effective": [
                r"\bmoderately effective ventricular emptying\b",
                r"\bmoderately effective emptying\b",
            ],

            "limited": [
                r"\blimited ventricular emptying\b",
                r"\blimited emptying\b",
            ],

            "generic_emptying": [
                r"\bemptying\b",
            ],
        },
    },


    # --------------------------------------------------------
    # 8. LV SYSTOLIC FUNCTION
    # --------------------------------------------------------

    "lv_systolic_function": {

        "presence": [
            r"\bpreserved left ventricular systolic function\b",
            r"\bmildly reduced left ventricular systolic function\b",
            r"\breduced left ventricular systolic function\b",
            r"\bleft ventricular systolic function\b",
        ],

        "values": {

            "preserved": [
                r"\bpreserved left ventricular systolic function\b",
            ],

            "mildly_reduced": [
                r"\bmildly reduced left ventricular systolic function\b",
            ],

            "reduced": [
                r"\breduced left ventricular systolic function\b",
            ],

            "generic_systolic_function": [
                r"\bleft ventricular systolic function\b",
            ],
        },
    },


    # --------------------------------------------------------
    # 9. PUMPING PERFORMANCE
    # --------------------------------------------------------

    "pumping_performance": {

        "presence": [
            r"\bgood pumping performance\b",
            r"\bmoderate pumping performance\b",
            r"\breduced pumping performance\b",
            r"\bpumping performance\b",
        ],

        "values": {

            "good": [
                r"\bgood pumping performance\b",
            ],

            "moderate": [
                r"\bmoderate pumping performance\b",
            ],

            "reduced": [
                r"\breduced pumping performance\b",
            ],

            "generic_pumping": [
                r"\bpumping performance\b",
            ],
        },
    },
}


# ============================================================
# CLINICAL EXTRACTION
# ============================================================

def field_present(text, patterns):

    text = normalize_text(text)

    return any(
        re.search(pattern, text)
        for pattern in patterns
    )


def find_first_value(
    text,
    value_patterns
):

    text = normalize_text(text)

    for value_name, patterns in value_patterns.items():

        for pattern in patterns:

            if re.search(
                pattern,
                text
            ):

                return value_name

    return None


def extract_clinical_state(
    text
):

    state = {}

    for field_name, config in FIELD_CONFIG.items():

        present = field_present(
            text,
            config["presence"]
        )

        value = None

        if present:

            value = find_first_value(
                text,
                config["values"]
            )

        state[field_name] = {
            "present": int(present),
            "value": value,
        }

    return state


# ============================================================
# ATTRIBUTE AGREEMENT
# ============================================================

def get_attribute_agreement(
    field_name,
    gt_value,
    pred_value
):

    # No GT attribute -> not evaluated
    if gt_value is None:

        return np.nan

    # Prediction missed the field/value
    if pred_value is None:

        return 0.0

    # Exact match
    if gt_value == pred_value:

        return 1.0


    # --------------------------------------------------------
    # Diastolic filling
    #
    # relatively_limited < moderate < substantial
    # --------------------------------------------------------

    if field_name == "diastolic_filling":

        severity_rank = {

            "relatively_limited": 0,

            "moderate": 1,

            "substantial": 2,
        }

        if (
            gt_value in severity_rank
            and pred_value in severity_rank
        ):

            distance = abs(
                severity_rank[gt_value]
                -
                severity_rank[pred_value]
            )

            max_distance = 2

            return (
                1.0
                -
                distance / max_distance
            )

        return 0.0


    # --------------------------------------------------------
    # Systolic cavity reduction
    #
    # moderate < marked
    #
    # "reduced" or "decreased" is treated as
    # partially matching a specific reduction level.
    # --------------------------------------------------------

    if field_name == "systolic_cavity_reduction":

        severity_rank = {

            "moderate": 0,

            "marked": 1,
        }

        if (
            gt_value in severity_rank
            and pred_value in severity_rank
        ):

            distance = abs(
                severity_rank[gt_value]
                -
                severity_rank[pred_value]
            )

            return (
                1.0
                -
                distance
            )

        if (
            gt_value in {
                "moderate",
                "marked"
            }
            and
            pred_value in {
                "reduced",
                "decreased"
            }
        ):

            return 0.5

        return 0.0


    # --------------------------------------------------------
    # Post-contraction cavity
    #
    # small < moderate < substantial
    # --------------------------------------------------------

    if field_name == "post_contraction_cavity":

        severity_rank = {

            "small": 0,

            "moderate": 1,

            "substantial": 2,
        }

        if (
            gt_value in severity_rank
            and pred_value in severity_rank
        ):

            distance = abs(
                severity_rank[gt_value]
                -
                severity_rank[pred_value]
            )

            return (
                1.0
                -
                distance / 2
            )

        return 0.0


    # --------------------------------------------------------
    # Ventricular emptying
    #
    # limited < moderately_effective < effective
    # --------------------------------------------------------

    if field_name == "ventricular_emptying":

        severity_rank = {

            "limited": 0,

            "moderately_effective": 1,

            "effective": 2,
        }

        if (
            gt_value in severity_rank
            and pred_value in severity_rank
        ):

            distance = abs(
                severity_rank[gt_value]
                -
                severity_rank[pred_value]
            )

            return (
                1.0
                -
                distance / 2
            )

        return 0.0


    # --------------------------------------------------------
    # LV systolic function
    #
    # preserved / mildly reduced / reduced
    # --------------------------------------------------------

    if field_name == "lv_systolic_function":

        severity_rank = {

            "preserved": 2,

            "mildly_reduced": 1,

            "reduced": 0,
        }

        if (
            gt_value in severity_rank
            and pred_value in severity_rank
        ):

            distance = abs(
                severity_rank[gt_value]
                -
                severity_rank[pred_value]
            )

            return (
                1.0
                -
                distance / 2
            )

        return 0.0


    # --------------------------------------------------------
    # Pumping performance
    #
    # reduced < moderate < good
    # --------------------------------------------------------

    if field_name == "pumping_performance":

        severity_rank = {

            "reduced": 0,

            "moderate": 1,

            "good": 2,
        }

        if (
            gt_value in severity_rank
            and pred_value in severity_rank
        ):

            distance = abs(
                severity_rank[gt_value]
                -
                severity_rank[pred_value]
            )

            return (
                1.0
                -
                distance / 2
            )

        return 0.0


    # --------------------------------------------------------
    # Categorical fields
    # --------------------------------------------------------

    if field_name == "cardiac_structure":

        if (
            gt_value == pred_value
        ):

            return 1.0

        # General heart mention when GT specifies
        # left ventricle is considered partial agreement.
        if (
            gt_value == "left_ventricle"
            and pred_value == "heart"
        ):

            return 0.5

        return 0.0


    if field_name in {
        "diastolic_phase",
        "systolic_phase"
    }:

        return 0.0


    return 0.0


# ============================================================
# CLINICAL SCORES
# ============================================================

def clinical_scores(
    gt,
    prediction
):

    gt_state = extract_clinical_state(gt)

    pred_state = extract_clinical_state(
        prediction
    )

    field_hits = []

    exact_attribute_hits = []

    agreement_scores = []

    details = []


    for field_name in FIELD_CONFIG.keys():

        gt_item = gt_state[field_name]

        pred_item = pred_state[field_name]


        # Only evaluate fields present in GT
        if gt_item["present"] != 1:

            continue


        # ----------------------------------------------------
        # Field recall
        # ----------------------------------------------------

        field_hit = int(
            pred_item["present"] == 1
        )

        field_hits.append(
            field_hit
        )


        # ----------------------------------------------------
        # Exact attribute accuracy
        # ----------------------------------------------------

        gt_value = gt_item["value"]

        pred_value = pred_item["value"]

        if gt_value is not None:

            exact_hit = int(
                pred_value == gt_value
            )

            exact_attribute_hits.append(
                exact_hit
            )

            agreement = get_attribute_agreement(
                field_name,
                gt_value,
                pred_value
            )

            agreement_scores.append(
                agreement
            )

        else:

            exact_hit = np.nan

            agreement = np.nan


        details.append({

            "Field":
                field_name,

            "GT Value":
                gt_value,

            "Prediction Value":
                pred_value,

            "Field Detected":
                field_hit,

            "Exact Attribute Correct":
                exact_hit,

            "Attribute Agreement":
                agreement,
        })


    field_recall = (

        float(
            np.mean(field_hits)
        )

        if field_hits

        else np.nan
    )


    exact_attribute_accuracy = (

        float(
            np.mean(
                exact_attribute_hits
            )
        )

        if exact_attribute_hits

        else np.nan
    )


    attribute_agreement = (

        float(
            np.mean(
                agreement_scores
            )
        )

        if agreement_scores

        else np.nan
    )


    return (
        field_recall,
        exact_attribute_accuracy,
        attribute_agreement,
        details,
    )


# ============================================================
# MODEL-LEVEL EVALUATION
# ============================================================

def evaluate_model(
    samples,
    prediction_key
):

    references = [
        sample["gt"]
        for sample in samples
    ]

    predictions = [
        sample[prediction_key]
        for sample in samples
    ]


    # Caption metrics
    bleu = calculate_bleu(
        references,
        predictions
    )

    rouge_l = calculate_rouge_l(
        references,
        predictions
    )

    meteor = calculate_meteor(
        references,
        predictions
    )

    cider = calculate_cider(
        references,
        predictions
    )


    # Content metrics
    field_scores = []

    exact_scores = []

    agreement_scores = []


    for gt, pred in zip(
        references,
        predictions
    ):

        (
            field_recall,
            exact_attribute_accuracy,
            attribute_agreement,
            _
        ) = clinical_scores(
            gt,
            pred
        )


        if not np.isnan(field_recall):

            field_scores.append(
                field_recall
            )


        if not np.isnan(
            exact_attribute_accuracy
        ):

            exact_scores.append(
                exact_attribute_accuracy
            )


        if not np.isnan(
            attribute_agreement
        ):

            agreement_scores.append(
                attribute_agreement
            )


    clinical_field_recall = (

        float(
            np.mean(field_scores)
        )

        if field_scores

        else np.nan
    )


    clinical_exact_accuracy = (

        float(
            np.mean(exact_scores)
        )

        if exact_scores

        else np.nan
    )


    clinical_agreement = (

        float(
            np.mean(agreement_scores)
        )

        if agreement_scores

        else np.nan
    )


    return {

        "BLEU-1":
            bleu["BLEU-1"],

        "BLEU-2":
            bleu["BLEU-2"],

        "BLEU-3":
            bleu["BLEU-3"],

        "BLEU-4":
            bleu["BLEU-4"],

        "ROUGE-L":
            rouge_l,

        "METEOR":
            meteor,

        "CIDEr":
            cider,

        "Clinical Field Recall":
            clinical_field_recall,

        "Exact Clinical Attribute Accuracy":
            clinical_exact_accuracy,

        "Clinical Attribute Agreement":
            clinical_agreement,
    }


# ============================================================
# DETAILED CLINICAL TABLE
# ============================================================

def build_clinical_table(
    samples
):

    rows = []


    for sample in samples:

        for model_name, prediction_key in [

            (
                "VideoChat2",
                "before"
            ),

            (
                "VideoChat2 + GPEC",
                "after"
            ),
        ]:

            gt = sample["gt"]

            pred = sample[
                prediction_key
            ]


            (
                _,
                _,
                _,
                details
            ) = clinical_scores(
                gt,
                pred
            )


            for detail in details:

                rows.append({

                    "ID":
                        sample["id"],

                    "Model":
                        model_name,

                    "Field":
                        detail[
                            "Field"
                        ],

                    "GT Value":
                        detail[
                            "GT Value"
                        ],

                    "Prediction Value":
                        detail[
                            "Prediction Value"
                        ],

                    "Field Detected":
                        detail[
                            "Field Detected"
                        ],

                    "Exact Attribute Correct":
                        detail[
                            "Exact Attribute Correct"
                        ],

                    "Attribute Agreement":
                        detail[
                            "Attribute Agreement"
                        ],
                })


    return pd.DataFrame(rows)


# ============================================================
# PER-SAMPLE CAPTION + CONTENT METRICS
# ============================================================

def build_per_sample_metrics(
    samples
):

    rows = []

    rouge = rouge_scorer.RougeScorer(
        ["rougeL"],
        use_stemmer=True
    )


    for sample in samples:

        gt = sample["gt"]


        for model_name, prediction_key in [

            (
                "VideoChat2",
                "before"
            ),

            (
                "VideoChat2 + GPEC",
                "after"
            ),
        ]:

            prediction = sample[
                prediction_key
            ]


            # ROUGE-L
            rouge_result = rouge.score(
                normalize_text(gt),
                normalize_text(prediction)
            )

            rouge_l = (
                rouge_result[
                    "rougeL"
                ].fmeasure
            )


            # METEOR
            pred_tokens = tokenize(
                prediction
            )

            if pred_tokens:

                meteor = meteor_score(
                    [tokenize(gt)],
                    pred_tokens
                )

            else:

                meteor = 0.0


            # Content-oriented metrics
            (
                field_recall,
                exact_attribute_accuracy,
                attribute_agreement,
                _
            ) = clinical_scores(
                gt,
                prediction
            )


            rows.append({

                "ID":
                    sample["id"],

                "Model":
                    model_name,

                "ROUGE-L":
                    rouge_l,

                "METEOR":
                    meteor,

                "Clinical Field Recall":
                    field_recall,

                "Exact Clinical Attribute Accuracy":
                    exact_attribute_accuracy,

                "Clinical Attribute Agreement":
                    attribute_agreement,
            })


    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\nReading:",
        INPUT_FILE
    )

    samples = parse_caption_file(
        INPUT_FILE
    )

    print(
        f"Number of samples found: "
        f"{len(samples)}"
    )

    if not samples:

        raise RuntimeError(
            "No samples found. "
            "Check the TXT format."
        )


    # ========================================================
    # VideoChat2
    # ========================================================

    print(
        "\nEvaluating VideoChat2..."
    )

    before_results = evaluate_model(
        samples,
        "before"
    )


    # ========================================================
    # VideoChat2 + GPEC
    # ========================================================

    print(
        "\nEvaluating VideoChat2 + GPEC..."
    )

    after_results = evaluate_model(
        samples,
        "after"
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    summary = pd.DataFrame([

        {
            "Model":
                "VideoChat2",

            **before_results,
        },

        {
            "Model":
                "VideoChat2 + GPEC",

            **after_results,
        },
    ])


    print(
        "\n=============================="
    )

    print(
        "OVERALL RESULTS"
    )

    print(
        "=============================="
    )

    print(
        summary.to_string(
            index=False
        )
    )


    summary.to_csv(
        SUMMARY_CSV,
        index=False
    )


    # ========================================================
    # PER-SAMPLE
    # ========================================================

    per_sample_df = (
        build_per_sample_metrics(
            samples
        )
    )

    per_sample_df.to_csv(
        PER_SAMPLE_CSV,
        index=False
    )


    # ========================================================
    # DETAILED CONTENT
    # ========================================================

    clinical_df = (
        build_clinical_table(
            samples
        )
    )

    clinical_df.to_csv(
        CLINICAL_CSV,
        index=False
    )


    # ========================================================
    # OUTPUT
    # ========================================================

    print(
        "\nSaved:"
    )

    print(
        " -",
        SUMMARY_CSV
    )

    print(
        " -",
        PER_SAMPLE_CSV
    )

    print(
        " -",
        CLINICAL_CSV
    )


if __name__ == "__main__":
    main()
