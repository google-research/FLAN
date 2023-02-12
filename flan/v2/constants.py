# Copyright 2022 The FLAN Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Constants."""

import os
import seqio
import t5.data

CACHE_DIRS = [os.path.join(os.path.dirname(__file__), 'cached_tasks')]

DEFAULT_OUTPUT_FEATURES = {
    'inputs':
        seqio.Feature(
            vocabulary=t5.data.get_default_vocabulary(),
            add_eos=True,
            required=False),
    'targets':
        seqio.Feature(
            vocabulary=t5.data.get_default_vocabulary(), add_eos=True)
}


FEW_SHOT_MAX_LEN = 1920
SEQUENCE_LEN = {'inputs': 2048, 'targets': 512}

# All train tasks will be defined for each set of features.
TRAIN_TASK_SUFFIXES_AND_FEATURES = [
    ('', DEFAULT_OUTPUT_FEATURES),
]
TRAIN_TASK_SUFFIXES = [x for x, _ in TRAIN_TASK_SUFFIXES_AND_FEATURES]


# Task name to number of cases. This is used when mixing tasks.
# If you use SEQUENCE_LEN other than {'inputs': 2048, 'targets': 512}, then this
# number-of-cases list is not accurate.
T_NAME_TO_NUM_CASES = {
    'bool_q': {
        'zero_shot': 9220,
        'x_shot': 9220,
        'five_shot': 9220,
    },
    'rte': {
        'zero_shot': 2290,
        'x_shot': 2290,
        'five_shot': 2288,
    },
    'wsc': {
        'zero_shot': 500,
        'x_shot': 500,
        'five_shot': 500,
    },
    'wic': {
        'zero_shot': 5220,
        'x_shot': 5220,
        'five_shot': 5220,
    },
    'natural_questions': {
        'zero_shot': 87720,
        'x_shot': 87720,
        'five_shot': 87720,
    },
    'record': {
        'zero_shot': 100520,
        'x_shot': 100520,
        'five_shot': 100448,
    },
    'trivia_qa': {
        'zero_shot': 87420,
        'x_shot': 87420,
        'five_shot': 87420,
    },
    'arc_challenge': {
        'zero_shot': 910,
        'x_shot': 910,
        'five_shot': 912,
    },
    'arc_easy': {
        'zero_shot': 2040,
        'x_shot': 2040,
        'five_shot': 2040,
    },
    'math_dataset': {
        'zero_shot': 1999780,
        'x_shot': 1999780,
        'five_shot': 1999780,
    },
    'aeslc': {
        'zero_shot': 14430,
        'x_shot': 14275,
        'five_shot': 14275,
    },
    'cnn_dailymail': {
        'zero_shot': 287040,
        'x_shot': 198109,
        'five_shot': 179512,
    },
    'gigaword': {
        'zero_shot': 2044370,
        'x_shot': 2044370,
        'five_shot': 2044416,
    },
    'multi_news': {
        'zero_shot': 44960,
        'x_shot': 3118,
        'five_shot': 1927,
    },
    'newsroom': {
        'zero_shot': 994560,
        'x_shot': 646889,
        'five_shot': 646099,
    },
    'opinion_abstracts_rotten_tomatoes': {
        'zero_shot': 3130,
        'x_shot': 3130,
        'five_shot': 3130,
    },
    'opinion_abstracts_idebate': {
        'zero_shot': 1650,
        'x_shot': 1650,
        'five_shot': 1655,
    },
    'coqa': {
        'zero_shot': 7090,
        'x_shot': 7077,
        'five_shot': 7085,
    },
    'samsum': {
        'zero_shot': 14730,
        'x_shot': 14730,
        'five_shot': 14728,
    },
    'xsum': {
        'zero_shot': 203560,
        'x_shot': 192545,
        'five_shot': 190400,
    },
    'squad_v1': {
        'zero_shot': 87390,
        'x_shot': 87390,
        'five_shot': 87397,
    },
    'squad_v2': {
        'zero_shot': 130110,
        'x_shot': 130109,
        'five_shot': 130110,
    },
    'drop': {
        'zero_shot': 77200,
        'x_shot': 77021,
        'five_shot': 77015,
    },
    'quac': {
        'zero_shot': 83360,
        'x_shot': 79325,
        'five_shot': 79290,
    },
    'multirc': {
        'zero_shot': 27040,
        'x_shot': 27040,
        'five_shot': 27042,
    },
    'ag_news_subset': {
        'zero_shot': 119800,
        'x_shot': 119800,
        'five_shot': 119798,
    },
    'anli_r1': {
        'zero_shot': 16940,
        'x_shot': 16940,
        'five_shot': 16944,
    },
    'anli_r2': {
        'zero_shot': 45460,
        'x_shot': 45460,
        'five_shot': 45456,
    },
    'anli_r3': {
        'zero_shot': 100450,
        'x_shot': 100450,
        'five_shot': 100456,
    },
    'sentiment140': {
        'zero_shot': 1599800,
        'x_shot': 1599800,
        'five_shot': 1599780,
    },
    'story_cloze': {
        'zero_shot': 1670,
        'x_shot': 1670,
        'five_shot': 1670,
    },
    'imdb_reviews': {
        'zero_shot': 24800,
        'x_shot': 24761,
        'five_shot': 24791,
    },
    'paws_wiki': {
        'zero_shot': 49400,
        'x_shot': 49400,
        'five_shot': 49400,
    },
    'definite_pronoun_resolution': {
        'zero_shot': 1120,
        'x_shot': 1120,
        'five_shot': 1120,
    },
    'glue_mrpc': {
        'zero_shot': 3460,
        'x_shot': 3460,
        'five_shot': 3465,
    },
    'glue_qqp': {
        'zero_shot': 363640,
        'x_shot': 363640,
        'five_shot': 363640,
    },
    'copa': {
        'zero_shot': 350,
        'x_shot': 350,
        'five_shot': 348,
    },
    'winogrande': {
        'zero_shot': 40190,
        'x_shot': 40190,
        'five_shot': 40194,
    },
    'yelp_polarity_reviews': {
        'zero_shot': 559800,
        'x_shot': 559774,
        'five_shot': 559798,
    },
    'cosmos_qa': {
        'zero_shot': 25060,
        'x_shot': 25060,
        'five_shot': 25060,
    },
    'para_crawl_enes': {
        'zero_shot': 30000,
        'x_shot': 30000,
        'five_shot': 29997,
    },
    'wmt14_enfr': {
        'zero_shot': 40835840,
        'x_shot': 40827181,
        'five_shot': 40827169,
    },
    'wmt16_translate_deen': {
        'zero_shot': 4548800,
        'x_shot': 4548675,
        'five_shot': 4548715,
    },
    'wmt16_translate_tren': {
        'zero_shot': 205750,
        'x_shot': 205750,
        'five_shot': 205752,
    },
    'wmt16_translate_csen': {
        'zero_shot': 52335360,
        'x_shot': 52325532,
        'five_shot': 52325751,
    },
    'wmt16_translate_fien': {
        'zero_shot': 2073360,
        'x_shot': 2073357,
        'five_shot': 2073342,
    },
    'wmt16_translate_roen': {
        'zero_shot': 610320,
        'x_shot': 610320,
        'five_shot': 610320,
    },
    'wmt16_translate_ruen': {
        'zero_shot': 2516160,
        'x_shot': 2516137,
        'five_shot': 2516133,
    },
    'common_gen': {
        'zero_shot': 67180,
        'x_shot': 67180,
        'five_shot': 67185,
    },
    'dart': {
        'zero_shot': 62450,
        'x_shot': 62450,
        'five_shot': 62454,
    },
    'e2e_nlg': {
        'zero_shot': 33320,
        'x_shot': 33320,
        'five_shot': 33320,
    },
    'web_nlg_en': {
        'zero_shot': 35220,
        'x_shot': 35220,
        'five_shot': 35224,
    },
    'wiki_lingua_english_en': {
        'zero_shot': 99000,
        'x_shot': 96918,
        'five_shot': 96486,
    },
    'true_case': {
        'zero_shot': 29260,
        'x_shot': 29260,
        'five_shot': 29260,
    },
    'fix_punct': {
        'zero_shot': 28070,
        'x_shot': 28070,
        'five_shot': 28070,
    },
    'word_segment': {
        'zero_shot': 30000,
        'x_shot': 30000,
        'five_shot': 29995,
    },
    'cb': {
        'zero_shot': 200,
        'x_shot': 200,
        'five_shot': 200,
    },
    'cola': {
        'zero_shot': 8350,
        'x_shot': 8350,
        'five_shot': 8348,
    },
    'sst2': {
        'zero_shot': 67140,
        'x_shot': 67140,
        'five_shot': 67148,
    },
    'mnli_matched': {
        'zero_shot': 392500,
        'x_shot': 392500,
        'five_shot': 392499,
    },
    'mnli_mismatched': {
        'zero_shot': 392500,
        'x_shot': 392500,
        'five_shot': 392499,
    },
    'qnli': {
        'zero_shot': 104540,
        'x_shot': 104540,
        'five_shot': 104536,
    },
    'wnli': {
        'zero_shot': 600,
        'x_shot': 600,
        'five_shot': 603,
    },
    'snli': {
        'zero_shot': 549360,
        'x_shot': 549360,
        'five_shot': 549360,
    },
    'trec': {
        'zero_shot': 5250,
        'x_shot': 5250,
        'five_shot': 5248,
    },
    'stsb': {
        'zero_shot': 5640,
        'x_shot': 5640,
        'five_shot': 5646,
    },
    'piqa': {
        'zero_shot': 16010,
        'x_shot': 16010,
        'five_shot': 16009,
    },
    'openbookqa': {
        'zero_shot': 4950,
        'x_shot': 4950,
        'five_shot': 4956,
    },
    'hellaswag': {
        'zero_shot': 39700,
        'x_shot': 39700,
        'five_shot': 39699,
    },
    'lambada': {
        'zero_shot': 4860,
        'x_shot': 4860,
        'five_shot': 4860,
    },
    'unified_qa_science_inst': {
        'zero_shot': 590,
        'x_shot': 590,
        'five_shot': 590,
    },
    'cot_gsm8k': {
        'zero_shot': 7470,
        'x_shot': 7470,
        'five_shot': 7473,
    },
    'cot_strategyqa': {
        'zero_shot': 2060,
        'x_shot': 2060,
        'five_shot': 2061,
    },
    'cot_creak': {
        'zero_shot': 6910,
        'x_shot': 6910,
        'five_shot': 6914,
    },
    'cot_qasc': {
        'zero_shot': 1080,
        'x_shot': 1080,
        'five_shot': 1084,
    },
    'cot_esnli': {
        'zero_shot': 36170,
        'x_shot': 36170,
        'five_shot': 36174,
    },
    'cot_ecqa': {
        'zero_shot': 7110,
        'x_shot': 7110,
        'five_shot': 7112,
    },
    'cot_sensemaking': {
        'zero_shot': 6070,
        'x_shot': 6070,
        'five_shot': 6069,
    },
    'stream_aqua': {
        'zero_shot': 2715,
        'x_shot': 2720,
        'five_shot': 2728,
    },
    'stream_qed': {
        'zero_shot': 5145,
        'x_shot': 5140,
        'five_shot': 5145,
    },
    'wiki_dialog': {
        'zero_shot': 11264000,
        'x_shot': 11263998,
        'five_shot': 11263997,
    },
    'qrecc': {
        'zero_shot': 10170,
        'x_shot': 10170,
        'five_shot': 10170,
    },
    't0_task_adaptation:adversarial_qa_dbert_answer_the_following_q': {
        'zero_shot': 10000,
        'x_shot': 10000,
        'five_shot': 9999,
    },
    't0_task_adaptation:adversarial_qa_dbert_based_on': {
        'zero_shot': 10000,
        'x_shot': 10000,
        'five_shot': 9999,
    },
    't0_task_adaptation:adversarial_qa_dbert_generate_question': {
        'zero_shot': 10000,
        'x_shot': 10000,
        'five_shot': 9999,
    },
    't0_task_adaptation:adversarial_qa_dbert_question_context_answer': {
        'zero_shot': 10000,
        'x_shot': 10000,
        'five_shot': 9999,
    },
    't0_task_adaptation:adversarial_qa_dbert_tell_what_it_is': {
        'zero_shot': 10000,
        'x_shot': 10000,
        'five_shot': 9999,
    },
    't0_task_adaptation:adversarial_qa_dbidaf_answer_the_following_q': {
        'zero_shot': 10000,
        'x_shot': 10000,
        'five_shot': 9999,
    },
    't0_task_adaptation:adversarial_qa_dbidaf_based_on': {
        'zero_shot': 10000,
        'x_shot': 10000,
        'five_shot': 9999,
    },
    't0_task_adaptation:adversarial_qa_dbidaf_generate_question': {
        'zero_shot': 10000,
        'x_shot': 10000,
        'five_shot': 9999,
    },
    't0_task_adaptation:adversarial_qa_dbidaf_question_context_answer': {
        'zero_shot': 10000,
        'x_shot': 10000,
        'five_shot': 9999,
    },
    't0_task_adaptation:adversarial_qa_dbidaf_tell_what_it_is': {
        'zero_shot': 10000,
        'x_shot': 10000,
        'five_shot': 9999,
    },
    't0_task_adaptation:adversarial_qa_droberta_answer_the_following_q': {
        'zero_shot': 10000,
        'x_shot': 10000,
        'five_shot': 9999,
    },
    't0_task_adaptation:adversarial_qa_droberta_based_on': {
        'zero_shot': 10000,
        'x_shot': 10000,
        'five_shot': 9999,
    },
    't0_task_adaptation:adversarial_qa_droberta_generate_question': {
        'zero_shot': 10000,
        'x_shot': 10000,
        'five_shot': 9999,
    },
    't0_task_adaptation:adversarial_qa_droberta_question_context_answer': {
        'zero_shot': 10000,
        'x_shot': 10000,
        'five_shot': 9999,
    },
    't0_task_adaptation:adversarial_qa_droberta_tell_what_it_is': {
        'zero_shot': 10000,
        'x_shot': 10000,
        'five_shot': 9999,
    },
    't0_task_adaptation:amazon_polarity_Is_this_product_review_positive': {
        'zero_shot': 3600000,
        'x_shot': 3600000,
        'five_shot': 3600000,
    },
    't0_task_adaptation:amazon_polarity_Is_this_review': {
        'zero_shot': 3600000,
        'x_shot': 3600000,
        'five_shot': 3600000,
    },
    't0_task_adaptation:amazon_polarity_Is_this_review_negative': {
        'zero_shot': 3600000,
        'x_shot': 3600000,
        'five_shot': 3600000,
    },
    't0_task_adaptation:amazon_polarity_User_recommend_this_product': {
        'zero_shot': 3600000,
        'x_shot': 3600000,
        'five_shot': 3600000,
    },
    't0_task_adaptation:amazon_polarity_convey_negative_or_positive_sentiment': {
        'zero_shot': 3600000,
        'x_shot': 3600000,
        'five_shot': 3600000,
    },
    't0_task_adaptation:amazon_polarity_flattering_or_not': {
        'zero_shot': 3600000,
        'x_shot': 3600000,
        'five_shot': 3600000,
    },
    't0_task_adaptation:amazon_polarity_negative_or_positive_tone': {
        'zero_shot': 3600000,
        'x_shot': 3600000,
        'five_shot': 3600000,
    },
    't0_task_adaptation:amazon_polarity_user_satisfied': {
        'zero_shot': 3600000,
        'x_shot': 3600000,
        'five_shot': 3600000,
    },
    't0_task_adaptation:amazon_polarity_would_you_buy': {
        'zero_shot': 3600000,
        'x_shot': 3600000,
        'five_shot': 3600000,
    },
    't0_task_adaptation:app_reviews_categorize_rating_using_review': {
        'zero_shot': 288060,
        'x_shot': 288060,
        'five_shot': 288060,
    },
    't0_task_adaptation:app_reviews_convert_to_rating': {
        'zero_shot': 288060,
        'x_shot': 288060,
        'five_shot': 288063,
    },
    't0_task_adaptation:app_reviews_convert_to_star_rating': {
        'zero_shot': 288060,
        'x_shot': 288060,
        'five_shot': 288060,
    },
    't0_task_adaptation:app_reviews_generate_review': {
        'zero_shot': 288060,
        'x_shot': 288060,
        'five_shot': 288063,
    },
    't0_task_adaptation:dbpedia_14_given_a_choice_of_categories_': {
        'zero_shot': 560000,
        'x_shot': 560000,
        'five_shot': 560000,
    },
    't0_task_adaptation:dbpedia_14_given_a_list_of_category_what_does_the_title_belong_to': {
        'zero_shot': 560000,
        'x_shot': 560000,
        'five_shot': 560000,
    },
    't0_task_adaptation:dbpedia_14_given_list_what_category_does_the_paragraph_belong_to': {
        'zero_shot': 560000,
        'x_shot': 560000,
        'five_shot': 560000,
    },
    't0_task_adaptation:dbpedia_14_pick_one_category_for_the_following_text': {
        'zero_shot': 560000,
        'x_shot': 560000,
        'five_shot': 560000,
    },
    't0_task_adaptation:dream_answer_to_dialogue': {
        'zero_shot': 6110,
        'x_shot': 6110,
        'five_shot': 6111,
    },
    't0_task_adaptation:dream_baseline': {
        'zero_shot': 6110,
        'x_shot': 6110,
        'five_shot': 6110,
    },
    't0_task_adaptation:dream_generate_first_utterance': {
        'zero_shot': 6110,
        'x_shot': 6110,
        'five_shot': 6111,
    },
    't0_task_adaptation:dream_generate_last_utterance': {
        'zero_shot': 6110,
        'x_shot': 6110,
        'five_shot': 6111,
    },
    't0_task_adaptation:dream_read_the_following_conversation_and_answer_the_question': {
        'zero_shot': 6110,
        'x_shot': 6110,
        'five_shot': 6110,
    },
    't0_task_adaptation:duorc_ParaphraseRC_answer_question': {
        'zero_shot': 69520,
        'x_shot': 69520,
        'five_shot': 69516,
    },
    't0_task_adaptation:duorc_ParaphraseRC_build_story_around_qa': {
        'zero_shot': 58750,
        'x_shot': 58750,
        'five_shot': 58752,
    },
    't0_task_adaptation:duorc_ParaphraseRC_decide_worth_it': {
        'zero_shot': 69520,
        'x_shot': 69520,
        'five_shot': 69516,
    },
    't0_task_adaptation:duorc_ParaphraseRC_extract_answer': {
        'zero_shot': 69520,
        'x_shot': 69520,
        'five_shot': 69516,
    },
    't0_task_adaptation:duorc_ParaphraseRC_generate_question': {
        'zero_shot': 58750,
        'x_shot': 58750,
        'five_shot': 58752,
    },
    't0_task_adaptation:duorc_ParaphraseRC_generate_question_by_answer': {
        'zero_shot': 58750,
        'x_shot': 58750,
        'five_shot': 58752,
    },
    't0_task_adaptation:duorc_ParaphraseRC_movie_director': {
        'zero_shot': 69520,
        'x_shot': 69520,
        'five_shot': 69516,
    },
    't0_task_adaptation:duorc_ParaphraseRC_question_answering': {
        'zero_shot': 69520,
        'x_shot': 69520,
        'five_shot': 69516,
    },
    't0_task_adaptation:duorc_ParaphraseRC_title_generation': {
        'zero_shot': 69520,
        'x_shot': 69520,
        'five_shot': 69516,
    },
    't0_task_adaptation:duorc_SelfRC_answer_question': {
        'zero_shot': 60720,
        'x_shot': 60720,
        'five_shot': 60714,
    },
    't0_task_adaptation:duorc_SelfRC_build_story_around_qa': {
        'zero_shot': 60090,
        'x_shot': 60090,
        'five_shot': 60093,
    },
    't0_task_adaptation:duorc_SelfRC_decide_worth_it': {
        'zero_shot': 60720,
        'x_shot': 60720,
        'five_shot': 60714,
    },
    't0_task_adaptation:duorc_SelfRC_extract_answer': {
        'zero_shot': 60720,
        'x_shot': 60720,
        'five_shot': 60714,
    },
    't0_task_adaptation:duorc_SelfRC_generate_question': {
        'zero_shot': 60090,
        'x_shot': 60090,
        'five_shot': 60093,
    },
    't0_task_adaptation:duorc_SelfRC_generate_question_by_answer': {
        'zero_shot': 60090,
        'x_shot': 60090,
        'five_shot': 60093,
    },
    't0_task_adaptation:duorc_SelfRC_movie_director': {
        'zero_shot': 60720,
        'x_shot': 60720,
        'five_shot': 60714,
    },
    't0_task_adaptation:duorc_SelfRC_question_answering': {
        'zero_shot': 60720,
        'x_shot': 60720,
        'five_shot': 60714,
    },
    't0_task_adaptation:duorc_SelfRC_title_generation': {
        'zero_shot': 60720,
        'x_shot': 60720,
        'five_shot': 60714,
    },
    't0_task_adaptation:kilt_tasks_hotpotqa_combining_facts': {
        'zero_shot': 88860,
        'x_shot': 88860,
        'five_shot': 88866,
    },
    't0_task_adaptation:kilt_tasks_hotpotqa_complex_question': {
        'zero_shot': 88860,
        'x_shot': 88860,
        'five_shot': 88866,
    },
    't0_task_adaptation:kilt_tasks_hotpotqa_final_exam': {
        'zero_shot': 88860,
        'x_shot': 88860,
        'five_shot': 88866,
    },
    't0_task_adaptation:kilt_tasks_hotpotqa_formulate': {
        'zero_shot': 88860,
        'x_shot': 88860,
        'five_shot': 88866,
    },
    't0_task_adaptation:kilt_tasks_hotpotqa_straighforward_qa': {
        'zero_shot': 88860,
        'x_shot': 88860,
        'five_shot': 88866,
    },
    't0_task_adaptation:qasc_is_correct_1': {
        'zero_shot': 8130,
        'x_shot': 8130,
        'five_shot': 8127,
    },
    't0_task_adaptation:qasc_is_correct_2': {
        'zero_shot': 8130,
        'x_shot': 8130,
        'five_shot': 8127,
    },
    't0_task_adaptation:qasc_qa_with_combined_facts_1': {
        'zero_shot': 8130,
        'x_shot': 8130,
        'five_shot': 8130,
    },
    't0_task_adaptation:qasc_qa_with_separated_facts_1': {
        'zero_shot': 8130,
        'x_shot': 8130,
        'five_shot': 8130,
    },
    't0_task_adaptation:qasc_qa_with_separated_facts_2': {
        'zero_shot': 8130,
        'x_shot': 8130,
        'five_shot': 8130,
    },
    't0_task_adaptation:qasc_qa_with_separated_facts_3': {
        'zero_shot': 8130,
        'x_shot': 8130,
        'five_shot': 8127,
    },
    't0_task_adaptation:qasc_qa_with_separated_facts_4': {
        'zero_shot': 8130,
        'x_shot': 8130,
        'five_shot': 8130,
    },
    't0_task_adaptation:qasc_qa_with_separated_facts_5': {
        'zero_shot': 8130,
        'x_shot': 8130,
        'five_shot': 8127,
    },
    't0_task_adaptation:quail_context_description_question_answer_id': {
        'zero_shot': 10240,
        'x_shot': 10240,
        'five_shot': 10240,
    },
    't0_task_adaptation:quail_context_description_question_answer_text': {
        'zero_shot': 10240,
        'x_shot': 10240,
        'five_shot': 10240,
    },
    't0_task_adaptation:quail_context_description_question_text': {
        'zero_shot': 10240,
        'x_shot': 10240,
        'five_shot': 10242,
    },
    't0_task_adaptation:quail_context_question_answer_description_id': {
        'zero_shot': 10240,
        'x_shot': 10240,
        'five_shot': 10240,
    },
    't0_task_adaptation:quail_context_question_answer_description_text': {
        'zero_shot': 10240,
        'x_shot': 10240,
        'five_shot': 10240,
    },
    't0_task_adaptation:quail_context_question_description_answer_id': {
        'zero_shot': 10240,
        'x_shot': 10240,
        'five_shot': 10240,
    },
    't0_task_adaptation:quail_context_question_description_answer_text': {
        'zero_shot': 10240,
        'x_shot': 10240,
        'five_shot': 10240,
    },
    't0_task_adaptation:quail_context_question_description_text': {
        'zero_shot': 10240,
        'x_shot': 10240,
        'five_shot': 10242,
    },
    't0_task_adaptation:quail_description_context_question_answer_id': {
        'zero_shot': 10240,
        'x_shot': 10240,
        'five_shot': 10240,
    },
    't0_task_adaptation:quail_description_context_question_answer_text': {
        'zero_shot': 10240,
        'x_shot': 10240,
        'five_shot': 10240,
    },
    't0_task_adaptation:quail_description_context_question_text': {
        'zero_shot': 10240,
        'x_shot': 10240,
        'five_shot': 10242,
    },
    't0_task_adaptation:quail_no_prompt_id': {
        'zero_shot': 10240,
        'x_shot': 10240,
        'five_shot': 10240,
    },
    't0_task_adaptation:quail_no_prompt_text': {
        'zero_shot': 10240,
        'x_shot': 10240,
        'five_shot': 10240,
    },
    't0_task_adaptation:quarel_choose_between': {
        'zero_shot': 1940,
        'x_shot': 1940,
        'five_shot': 1940,
    },
    't0_task_adaptation:quarel_do_not_use': {
        'zero_shot': 1940,
        'x_shot': 1940,
        'five_shot': 1940,
    },
    't0_task_adaptation:quarel_heres_a_story': {
        'zero_shot': 1940,
        'x_shot': 1940,
        'five_shot': 1940,
    },
    't0_task_adaptation:quarel_logic_test': {
        'zero_shot': 1940,
        'x_shot': 1940,
        'five_shot': 1940,
    },
    't0_task_adaptation:quarel_testing_students': {
        'zero_shot': 1940,
        'x_shot': 1940,
        'five_shot': 1940,
    },
    't0_task_adaptation:quartz_answer_question_based_on': {
        'zero_shot': 2690,
        'x_shot': 2690,
        'five_shot': 2690,
    },
    't0_task_adaptation:quartz_answer_question_below': {
        'zero_shot': 2690,
        'x_shot': 2690,
        'five_shot': 2690,
    },
    't0_task_adaptation:quartz_given_the_fact_answer_the_q': {
        'zero_shot': 2690,
        'x_shot': 2690,
        'five_shot': 2690,
    },
    't0_task_adaptation:quartz_having_read_above_passage': {
        'zero_shot': 2690,
        'x_shot': 2690,
        'five_shot': 2690,
    },
    't0_task_adaptation:quartz_paragraph_question_plain_concat': {
        'zero_shot': 2690,
        'x_shot': 2690,
        'five_shot': 2690,
    },
    't0_task_adaptation:quartz_read_passage_below_choose': {
        'zero_shot': 2690,
        'x_shot': 2690,
        'five_shot': 2690,
    },
    't0_task_adaptation:quartz_use_info_from_paragraph_question': {
        'zero_shot': 2690,
        'x_shot': 2690,
        'five_shot': 2690,
    },
    't0_task_adaptation:quartz_use_info_from_question_paragraph': {
        'zero_shot': 2690,
        'x_shot': 2690,
        'five_shot': 2690,
    },
    't0_task_adaptation:quoref_Answer_Friend_Question': {
        'zero_shot': 19390,
        'x_shot': 19390,
        'five_shot': 19395,
    },
    't0_task_adaptation:quoref_Answer_Question_Given_Context': {
        'zero_shot': 19390,
        'x_shot': 19390,
        'five_shot': 19395,
    },
    't0_task_adaptation:quoref_Answer_Test': {
        'zero_shot': 19390,
        'x_shot': 19390,
        'five_shot': 19395,
    },
    't0_task_adaptation:quoref_Context_Contains_Answer': {
        'zero_shot': 19390,
        'x_shot': 19390,
        'five_shot': 19395,
    },
    't0_task_adaptation:quoref_Find_Answer': {
        'zero_shot': 19390,
        'x_shot': 19390,
        'five_shot': 19395,
    },
    't0_task_adaptation:quoref_Found_Context_Online': {
        'zero_shot': 19390,
        'x_shot': 19390,
        'five_shot': 19395,
    },
    't0_task_adaptation:quoref_Given_Context_Answer_Question': {
        'zero_shot': 19390,
        'x_shot': 19390,
        'five_shot': 19395,
    },
    't0_task_adaptation:quoref_Guess_Answer': {
        'zero_shot': 19390,
        'x_shot': 19390,
        'five_shot': 19395,
    },
    't0_task_adaptation:quoref_Guess_Title_For_Context': {
        'zero_shot': 19390,
        'x_shot': 19390,
        'five_shot': 19395,
    },
    't0_task_adaptation:quoref_Read_And_Extract_': {
        'zero_shot': 19390,
        'x_shot': 19390,
        'five_shot': 19395,
    },
    't0_task_adaptation:quoref_What_Is_The_Answer': {
        'zero_shot': 19390,
        'x_shot': 19390,
        'five_shot': 19395,
    },
    't0_task_adaptation:race_high_Is_this_the_right_answer': {
        'zero_shot': 62440,
        'x_shot': 62440,
        'five_shot': 62440,
    },
    't0_task_adaptation:race_high_Read_the_article_and_answer_the_question_no_option_': {
        'zero_shot': 62440,
        'x_shot': 62440,
        'five_shot': 62442,
    },
    't0_task_adaptation:race_high_Select_the_best_answer': {
        'zero_shot': 62440,
        'x_shot': 62440,
        'five_shot': 62440,
    },
    't0_task_adaptation:race_high_Select_the_best_answer_generate_span_': {
        'zero_shot': 62440,
        'x_shot': 62440,
        'five_shot': 62440,
    },
    't0_task_adaptation:race_high_Select_the_best_answer_no_instructions_': {
        'zero_shot': 62440,
        'x_shot': 62440,
        'five_shot': 62440,
    },
    't0_task_adaptation:race_high_Taking_a_test': {
        'zero_shot': 62440,
        'x_shot': 62440,
        'five_shot': 62440,
    },
    't0_task_adaptation:race_high_Write_a_multi_choice_question_for_the_following_article': {
        'zero_shot': 62440,
        'x_shot': 62440,
        'five_shot': 62442,
    },
    't0_task_adaptation:race_high_Write_a_multi_choice_question_options_given_': {
        'zero_shot': 62440,
        'x_shot': 62440,
        'five_shot': 62442,
    },
    't0_task_adaptation:race_middle_Is_this_the_right_answer': {
        'zero_shot': 25420,
        'x_shot': 25420,
        'five_shot': 25420,
    },
    't0_task_adaptation:race_middle_Read_the_article_and_answer_the_question_no_option_': {
        'zero_shot': 25420,
        'x_shot': 25420,
        'five_shot': 25416,
    },
    't0_task_adaptation:race_middle_Select_the_best_answer': {
        'zero_shot': 25420,
        'x_shot': 25420,
        'five_shot': 25420,
    },
    't0_task_adaptation:race_middle_Select_the_best_answer_generate_span_': {
        'zero_shot': 25420,
        'x_shot': 25420,
        'five_shot': 25420,
    },
    't0_task_adaptation:race_middle_Select_the_best_answer_no_instructions_': {
        'zero_shot': 25420,
        'x_shot': 25420,
        'five_shot': 25420,
    },
    't0_task_adaptation:race_middle_Taking_a_test': {
        'zero_shot': 25420,
        'x_shot': 25420,
        'five_shot': 25420,
    },
    't0_task_adaptation:race_middle_Write_a_multi_choice_question_for_the_following_article': {
        'zero_shot': 25420,
        'x_shot': 25420,
        'five_shot': 25416,
    },
    't0_task_adaptation:race_middle_Write_a_multi_choice_question_options_given_': {
        'zero_shot': 25420,
        'x_shot': 25420,
        'five_shot': 25416,
    },
    't0_task_adaptation:ropes_background_new_situation_answer': {
        'zero_shot': 10920,
        'x_shot': 10920,
        'five_shot': 10917,
    },
    't0_task_adaptation:ropes_background_situation_middle': {
        'zero_shot': 10920,
        'x_shot': 10920,
        'five_shot': 10917,
    },
    't0_task_adaptation:ropes_given_background_situation': {
        'zero_shot': 10920,
        'x_shot': 10920,
        'five_shot': 10917,
    },
    't0_task_adaptation:ropes_new_situation_background_answer': {
        'zero_shot': 10920,
        'x_shot': 10920,
        'five_shot': 10917,
    },
    't0_task_adaptation:ropes_plain_background_situation': {
        'zero_shot': 10920,
        'x_shot': 10920,
        'five_shot': 10917,
    },
    't0_task_adaptation:ropes_plain_bottom_hint': {
        'zero_shot': 10920,
        'x_shot': 10920,
        'five_shot': 10917,
    },
    't0_task_adaptation:ropes_plain_no_background': {
        'zero_shot': 10920,
        'x_shot': 10920,
        'five_shot': 10917,
    },
    't0_task_adaptation:ropes_prompt_beginning': {
        'zero_shot': 10920,
        'x_shot': 10920,
        'five_shot': 10917,
    },
    't0_task_adaptation:ropes_prompt_bottom_hint_beginning': {
        'zero_shot': 10920,
        'x_shot': 10920,
        'five_shot': 10917,
    },
    't0_task_adaptation:ropes_prompt_bottom_no_hint': {
        'zero_shot': 10920,
        'x_shot': 10920,
        'five_shot': 10917,
    },
    't0_task_adaptation:ropes_prompt_mix': {
        'zero_shot': 10920,
        'x_shot': 10920,
        'five_shot': 10917,
    },
    't0_task_adaptation:ropes_read_background_situation': {
        'zero_shot': 10920,
        'x_shot': 10920,
        'five_shot': 10917,
    },
    't0_task_adaptation:sciq_Direct_Question': {
        'zero_shot': 11670,
        'x_shot': 11670,
        'five_shot': 11673,
    },
    't0_task_adaptation:sciq_Direct_Question_Closed_Book_': {
        'zero_shot': 11670,
        'x_shot': 11670,
        'five_shot': 11673,
    },
    't0_task_adaptation:sciq_Multiple_Choice': {
        'zero_shot': 11670,
        'x_shot': 11670,
        'five_shot': 11670,
    },
    't0_task_adaptation:sciq_Multiple_Choice_Closed_Book_': {
        'zero_shot': 11670,
        'x_shot': 11670,
        'five_shot': 11670,
    },
    't0_task_adaptation:sciq_Multiple_Choice_Question_First': {
        'zero_shot': 11670,
        'x_shot': 11670,
        'five_shot': 11670,
    },
    't0_task_adaptation:social_i_qa_Check_if_a_random_answer_is_valid_or_not': {
        'zero_shot': 33410,
        'x_shot': 33410,
        'five_shot': 33408,
    },
    't0_task_adaptation:social_i_qa_Generate_answer': {
        'zero_shot': 33410,
        'x_shot': 33410,
        'five_shot': 33408,
    },
    't0_task_adaptation:social_i_qa_Generate_the_question_from_the_answer': {
        'zero_shot': 33410,
        'x_shot': 33410,
        'five_shot': 33408,
    },
    't0_task_adaptation:social_i_qa_I_was_wondering': {
        'zero_shot': 33410,
        'x_shot': 33410,
        'five_shot': 33408,
    },
    't0_task_adaptation:social_i_qa_Show_choices_and_generate_answer': {
        'zero_shot': 33410,
        'x_shot': 33410,
        'five_shot': 33410,
    },
    't0_task_adaptation:social_i_qa_Show_choices_and_generate_index': {
        'zero_shot': 33410,
        'x_shot': 33410,
        'five_shot': 33410,
    },
    't0_task_adaptation:web_questions_get_the_answer': {
        'zero_shot': 3770,
        'x_shot': 3770,
        'five_shot': 3771,
    },
    't0_task_adaptation:web_questions_potential_correct_answer': {
        'zero_shot': 3770,
        'x_shot': 3770,
        'five_shot': 3771,
    },
    't0_task_adaptation:web_questions_question_answer': {
        'zero_shot': 3770,
        'x_shot': 3770,
        'five_shot': 3771,
    },
    't0_task_adaptation:web_questions_short_general_knowledge_q': {
        'zero_shot': 3770,
        'x_shot': 3770,
        'five_shot': 3771,
    },
    't0_task_adaptation:web_questions_whats_the_answer': {
        'zero_shot': 3770,
        'x_shot': 3770,
        'five_shot': 3771,
    },
    't0_task_adaptation:wiki_bio_comprehension': {
        'zero_shot': 582630,
        'x_shot': 582563,
        'five_shot': 582568,
    },
    't0_task_adaptation:wiki_bio_guess_person': {
        'zero_shot': 582630,
        'x_shot': 582630,
        'five_shot': 582633,
    },
    't0_task_adaptation:wiki_bio_key_content': {
        'zero_shot': 582630,
        'x_shot': 582601,
        'five_shot': 582610,
    },
    't0_task_adaptation:wiki_bio_what_content': {
        'zero_shot': 582630,
        'x_shot': 582630,
        'five_shot': 582633,
    },
    't0_task_adaptation:wiki_bio_who': {
        'zero_shot': 582630,
        'x_shot': 582610,
        'five_shot': 582616,
    },
    't0_task_adaptation:wiki_hop_original_choose_best_object_affirmative_1': {
        'zero_shot': 43730,
        'x_shot': 7569,
        'five_shot': 7541,
    },
    't0_task_adaptation:wiki_hop_original_choose_best_object_affirmative_2': {
        'zero_shot': 43730,
        'x_shot': 7505,
        'five_shot': 7522,
    },
    't0_task_adaptation:wiki_hop_original_choose_best_object_affirmative_3': {
        'zero_shot': 43730,
        'x_shot': 7350,
        'five_shot': 7398,
    },
    't0_task_adaptation:wiki_hop_original_choose_best_object_interrogative_1': {
        'zero_shot': 43730,
        'x_shot': 7679,
        'five_shot': 7713,
    },
    't0_task_adaptation:wiki_hop_original_choose_best_object_interrogative_2': {
        'zero_shot': 43730,
        'x_shot': 7766,
        'five_shot': 7799,
    },
    't0_task_adaptation:wiki_hop_original_explain_relation': {
        'zero_shot': 43730,
        'x_shot': 8907,
        'five_shot': 8778,
    },
    't0_task_adaptation:wiki_hop_original_generate_object': {
        'zero_shot': 43730,
        'x_shot': 8878,
        'five_shot': 8767,
    },
    't0_task_adaptation:wiki_hop_original_generate_subject': {
        'zero_shot': 43730,
        'x_shot': 8778,
        'five_shot': 8670,
    },
    't0_task_adaptation:wiki_hop_original_generate_subject_and_object': {
        'zero_shot': 43730,
        'x_shot': 8778,
        'five_shot': 8670,
    },
    't0_task_adaptation:wiki_qa_Decide_good_answer': {
        'zero_shot': 20360,
        'x_shot': 20360,
        'five_shot': 20360,
    },
    't0_task_adaptation:wiki_qa_Direct_Answer_to_Question': {
        'zero_shot': 1040,
        'x_shot': 1040,
        'five_shot': 1035,
    },
    't0_task_adaptation:wiki_qa_Generate_Question_from_Topic': {
        'zero_shot': 1040,
        'x_shot': 1040,
        'five_shot': 1035,
    },
    't0_task_adaptation:wiki_qa_Is_This_True_': {
        'zero_shot': 20360,
        'x_shot': 20360,
        'five_shot': 20358,
    },
    't0_task_adaptation:wiki_qa_Jeopardy_style': {
        'zero_shot': 1040,
        'x_shot': 1040,
        'five_shot': 1035,
    },
    't0_task_adaptation:wiki_qa_Topic_Prediction_Answer_Only': {
        'zero_shot': 1040,
        'x_shot': 1040,
        'five_shot': 1035,
    },
    't0_task_adaptation:wiki_qa_Topic_Prediction_Question_Only': {
        'zero_shot': 1040,
        'x_shot': 1040,
        'five_shot': 1035,
    },
    't0_task_adaptation:wiki_qa_Topic_Prediction_Question_and_Answer_Pair': {
        'zero_shot': 1040,
        'x_shot': 1040,
        'five_shot': 1035,
    },
    't0_task_adaptation:wiki_qa_automatic_system': {
        'zero_shot': 20360,
        'x_shot': 20360,
        'five_shot': 20358,
    },
    't0_task_adaptation:wiki_qa_exercise': {
        'zero_shot': 20360,
        'x_shot': 20360,
        'five_shot': 20360,
    },
    't0_task_adaptation:wiki_qa_found_on_google': {
        'zero_shot': 20360,
        'x_shot': 20360,
        'five_shot': 20360,
    },
    't0_task_adaptation:wiqa_does_the_supposed_perturbation_have_an_effect': {
        'zero_shot': 29800,
        'x_shot': 29800,
        'five_shot': 29808,
    },
    't0_task_adaptation:wiqa_effect_with_label_answer': {
        'zero_shot': 29800,
        'x_shot': 29800,
        'five_shot': 29808,
    },
    't0_task_adaptation:wiqa_effect_with_string_answer': {
        'zero_shot': 29800,
        'x_shot': 29800,
        'five_shot': 29808,
    },
    't0_task_adaptation:wiqa_what_is_the_final_step_of_the_following_process': {
        'zero_shot': 29800,
        'x_shot': 29800,
        'five_shot': 29808,
    },
    't0_task_adaptation:wiqa_what_is_the_missing_first_step': {
        'zero_shot': 29800,
        'x_shot': 29800,
        'five_shot': 29808,
    },
    't0_task_adaptation:wiqa_what_might_be_the_first_step_of_the_process': {
        'zero_shot': 29800,
        'x_shot': 29800,
        'five_shot': 29808,
    },
    't0_task_adaptation:wiqa_what_might_be_the_last_step_of_the_process': {
        'zero_shot': 29800,
        'x_shot': 29800,
        'five_shot': 29808,
    },
    't0_task_adaptation:wiqa_which_of_the_following_is_the_supposed_perturbation': {
        'zero_shot': 29800,
        'x_shot': 29800,
        'five_shot': 29808,
    },
    'cot_input_inversion_gsm8k': {
        'zero_shot': 7470,
        'x_shot': 7470,
        'five_shot': 7473,
    },
    'cot_input_inversion_strategyqa': {
        'zero_shot': 2060,
        'x_shot': 2060,
        'five_shot': 2061,
    },
    'cot_input_inversion_creak': {
        'zero_shot': 6910,
        'x_shot': 6910,
        'five_shot': 6914,
    },
    'cot_input_inversion_qasc': {
        'zero_shot': 1080,
        'x_shot': 1080,
        'five_shot': 1084,
    },
    'cot_input_inversion_esnli': {
        'zero_shot': 36170,
        'x_shot': 36170,
        'five_shot': 36174,
    },
    'cot_input_inversion_ecqa': {
        'zero_shot': 7110,
        'x_shot': 7110,
        'five_shot': 7112,
    },
    'cot_input_inversion_sensemaking': {
        'zero_shot': 6070,
        'x_shot': 6070,
        'five_shot': 6069,
    },
    'stream_input_inversion_aqua': {
        'zero_shot': 2715,
        'x_shot': 2720,
        'five_shot': 2728,
    },
    'stream_input_inversion_qed': {
        'zero_shot': 5145,
        'x_shot': 5140,
        'five_shot': 5145,
    },
    'wiki_dialog_input_inversion': {
        'zero_shot': 11264000,
        'x_shot': 11263998,
        'five_shot': 11263997,
    },
    'qrecc_input_inversion': {
        'zero_shot': 10170,
        'x_shot': 10170,
        'five_shot': 10170,
    },
    't0_task_adaptation:cos_e_v1.11_aligned_with_common_sense': {
        'zero_shot': 9740,
        'x_shot': 9740,
        'five_shot': 9738,
    },
    't0_task_adaptation:cos_e_v1.11_description_question_option_id': {
        'zero_shot': 9740,
        'x_shot': 9740,
        'five_shot': 9740,
    },
    't0_task_adaptation:cos_e_v1.11_description_question_option_text': {
        'zero_shot': 9740,
        'x_shot': 9740,
        'five_shot': 9740,
    },
    't0_task_adaptation:cos_e_v1.11_explain_why_human': {
        'zero_shot': 9740,
        'x_shot': 9740,
        'five_shot': 9738,
    },
    't0_task_adaptation:cos_e_v1.11_generate_explanation_given_text': {
        'zero_shot': 9740,
        'x_shot': 9740,
        'five_shot': 9738,
    },
    't0_task_adaptation:cos_e_v1.11_i_think': {
        'zero_shot': 9740,
        'x_shot': 9740,
        'five_shot': 9738,
    },
    't0_task_adaptation:cos_e_v1.11_question_description_option_id': {
        'zero_shot': 9740,
        'x_shot': 9740,
        'five_shot': 9740,
    },
    't0_task_adaptation:cos_e_v1.11_question_description_option_text': {
        'zero_shot': 9740,
        'x_shot': 9740,
        'five_shot': 9740,
    },
    't0_task_adaptation:cos_e_v1.11_question_option_description_id': {
        'zero_shot': 9740,
        'x_shot': 9740,
        'five_shot': 9740,
    },
    't0_task_adaptation:cos_e_v1.11_question_option_description_text': {
        'zero_shot': 9740,
        'x_shot': 9740,
        'five_shot': 9740,
    },
    't0_task_adaptation:cos_e_v1.11_rationale': {
        'zero_shot': 9740,
        'x_shot': 9740,
        'five_shot': 9738,
    },
    'tfds_natural_instructions': {
        'zero_shot': 5_000_000,
        'x_shot': 5_000_000,
        'five_shot': 5_000_000,
    },
}
