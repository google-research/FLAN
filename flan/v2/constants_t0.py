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

"""Constants related to T0."""

T0_TASK_PREFIX = 't0_task_adaptation:'
T0_TRAIN_TASK_SPLITS = {
    't0_task_adaptation:adversarial_qa_dbert_answer_the_following_q': {
        'train': 10000,
        'validation': 1000
    },
    't0_task_adaptation:adversarial_qa_dbert_based_on': {
        'train': 10000,
        'validation': 1000
    },
    't0_task_adaptation:adversarial_qa_dbert_generate_question': {
        'test': 1000,
        'train': 10000,
        'validation': 1000
    },
    't0_task_adaptation:adversarial_qa_dbert_question_context_answer': {
        'train': 10000,
        'validation': 1000
    },
    't0_task_adaptation:adversarial_qa_dbert_tell_what_it_is': {
        'train': 10000,
        'validation': 1000
    },
    't0_task_adaptation:adversarial_qa_dbidaf_answer_the_following_q': {
        'train': 10000,
        'validation': 1000
    },
    't0_task_adaptation:adversarial_qa_dbidaf_based_on': {
        'train': 10000,
        'validation': 1000
    },
    't0_task_adaptation:adversarial_qa_dbidaf_generate_question': {
        'test': 1000,
        'train': 10000,
        'validation': 1000
    },
    't0_task_adaptation:adversarial_qa_dbidaf_question_context_answer': {
        'train': 10000,
        'validation': 1000
    },
    't0_task_adaptation:adversarial_qa_dbidaf_tell_what_it_is': {
        'train': 10000,
        'validation': 1000
    },
    't0_task_adaptation:adversarial_qa_droberta_answer_the_following_q': {
        'train': 10000,
        'validation': 1000
    },
    't0_task_adaptation:adversarial_qa_droberta_based_on': {
        'train': 10000,
        'validation': 1000
    },
    't0_task_adaptation:adversarial_qa_droberta_generate_question': {
        'test': 1000,
        'train': 10000,
        'validation': 1000
    },
    't0_task_adaptation:adversarial_qa_droberta_question_context_answer': {
        'train': 10000,
        'validation': 1000
    },
    't0_task_adaptation:adversarial_qa_droberta_tell_what_it_is': {
        'train': 10000,
        'validation': 1000
    },
    't0_task_adaptation:ag_news_classify': {
        'test': 7600,
        'train': 120000
    },
    't0_task_adaptation:ag_news_classify_question_first': {
        'test': 7600,
        'train': 120000
    },
    't0_task_adaptation:ag_news_classify_with_choices': {
        'test': 7600,
        'train': 120000
    },
    't0_task_adaptation:ag_news_classify_with_choices_question_first': {
        'test': 7600,
        'train': 120000
    },
    't0_task_adaptation:ag_news_recommend': {
        'test': 7600,
        'train': 120000
    },
    't0_task_adaptation:ag_news_which_section': {
        'test': 7600,
        'train': 120000
    },
    't0_task_adaptation:ag_news_which_section_choices': {
        'test': 7600,
        'train': 120000
    },
    't0_task_adaptation:ai2_arc_ARC_Challenge_heres_a_problem': {
        'test': 1172,
        'train': 1119,
        'validation': 299
    },
    't0_task_adaptation:ai2_arc_ARC_Challenge_i_am_hesitating': {
        'test': 1172,
        'train': 1119,
        'validation': 299
    },
    't0_task_adaptation:ai2_arc_ARC_Challenge_multiple_choice': {
        'test': 1172,
        'train': 1119,
        'validation': 299
    },
    't0_task_adaptation:ai2_arc_ARC_Challenge_pick_false_options': {
        'test': 1172,
        'train': 1119,
        'validation': 299
    },
    't0_task_adaptation:ai2_arc_ARC_Challenge_pick_the_most_correct_option': {
        'test': 1172,
        'train': 1119,
        'validation': 299
    },
    't0_task_adaptation:ai2_arc_ARC_Challenge_qa_options': {
        'test': 1172,
        'train': 1119,
        'validation': 299
    },
    't0_task_adaptation:ai2_arc_ARC_Easy_heres_a_problem': {
        'test': 2376,
        'train': 2251,
        'validation': 570
    },
    't0_task_adaptation:ai2_arc_ARC_Easy_i_am_hesitating': {
        'test': 2376,
        'train': 2251,
        'validation': 570
    },
    't0_task_adaptation:ai2_arc_ARC_Easy_multiple_choice': {
        'test': 2376,
        'train': 2251,
        'validation': 570
    },
    't0_task_adaptation:ai2_arc_ARC_Easy_pick_false_options': {
        'test': 2376,
        'train': 2251,
        'validation': 570
    },
    't0_task_adaptation:ai2_arc_ARC_Easy_pick_the_most_correct_option': {
        'test': 2376,
        'train': 2251,
        'validation': 570
    },
    't0_task_adaptation:ai2_arc_ARC_Easy_qa_options': {
        'test': 2376,
        'train': 2251,
        'validation': 570
    },
    't0_task_adaptation:amazon_polarity_Is_this_product_review_positive': {
        'test': 400000,
        'train': 3600000
    },
    't0_task_adaptation:amazon_polarity_Is_this_review': {
        'test': 400000,
        'train': 3600000
    },
    't0_task_adaptation:amazon_polarity_Is_this_review_negative': {
        'test': 400000,
        'train': 3600000
    },
    't0_task_adaptation:amazon_polarity_User_recommend_this_product': {
        'test': 400000,
        'train': 3600000
    },
    't0_task_adaptation:amazon_polarity_convey_negative_or_positive_sentiment':
        {
            'test': 400000,
            'train': 3600000
        },
    't0_task_adaptation:amazon_polarity_flattering_or_not': {
        'test': 400000,
        'train': 3600000
    },
    't0_task_adaptation:amazon_polarity_negative_or_positive_tone': {
        'test': 400000,
        'train': 3600000
    },
    't0_task_adaptation:amazon_polarity_user_satisfied': {
        'test': 400000,
        'train': 3600000
    },
    't0_task_adaptation:amazon_polarity_would_you_buy': {
        'test': 400000,
        'train': 3600000
    },
    't0_task_adaptation:anli_GPT_3_style_r1': {
        'test': 1000,
        'train': 16946,
        'validation': 1000
    },
    't0_task_adaptation:anli_GPT_3_style_r1_score_eval': {
        'test': 3000,
        'train': 50838,
        'validation': 3000
    },
    't0_task_adaptation:anli_GPT_3_style_r2': {
        'test': 1000,
        'train': 45460,
        'validation': 1000
    },
    't0_task_adaptation:anli_GPT_3_style_r2_score_eval': {
        'test': 3000,
        'train': 136380,
        'validation': 3000
    },
    't0_task_adaptation:anli_GPT_3_style_r3': {
        'test': 1200,
        'train': 100459,
        'validation': 1200
    },
    't0_task_adaptation:anli_GPT_3_style_r3_score_eval': {
        'test': 3600,
        'train': 301377,
        'validation': 3600
    },
    't0_task_adaptation:anli_MNLI_crowdsource_r1': {
        'test': 1000,
        'train': 16946,
        'validation': 1000
    },
    't0_task_adaptation:anli_MNLI_crowdsource_r1_score_eval': {
        'test': 3000,
        'train': 50838,
        'validation': 3000
    },
    't0_task_adaptation:anli_MNLI_crowdsource_r2': {
        'test': 1000,
        'train': 45460,
        'validation': 1000
    },
    't0_task_adaptation:anli_MNLI_crowdsource_r2_score_eval': {
        'test': 3000,
        'train': 136380,
        'validation': 3000
    },
    't0_task_adaptation:anli_MNLI_crowdsource_r3': {
        'test': 1200,
        'train': 100459,
        'validation': 1200
    },
    't0_task_adaptation:anli_MNLI_crowdsource_r3_score_eval': {
        'test': 3600,
        'train': 301377,
        'validation': 3600
    },
    't0_task_adaptation:anli_always_sometimes_never_r1': {
        'test': 1000,
        'train': 16946,
        'validation': 1000
    },
    't0_task_adaptation:anli_always_sometimes_never_r1_score_eval': {
        'test': 3000,
        'train': 50838,
        'validation': 3000
    },
    't0_task_adaptation:anli_always_sometimes_never_r2': {
        'test': 1000,
        'train': 45460,
        'validation': 1000
    },
    't0_task_adaptation:anli_always_sometimes_never_r2_score_eval': {
        'test': 3000,
        'train': 136380,
        'validation': 3000
    },
    't0_task_adaptation:anli_always_sometimes_never_r3': {
        'test': 1200,
        'train': 100459,
        'validation': 1200
    },
    't0_task_adaptation:anli_always_sometimes_never_r3_score_eval': {
        'test': 3600,
        'train': 301377,
        'validation': 3600
    },
    't0_task_adaptation:anli_based_on_the_previous_passage_r1': {
        'test': 1000,
        'train': 16946,
        'validation': 1000
    },
    't0_task_adaptation:anli_based_on_the_previous_passage_r1_score_eval': {
        'test': 3000,
        'train': 50838,
        'validation': 3000
    },
    't0_task_adaptation:anli_based_on_the_previous_passage_r2': {
        'test': 1000,
        'train': 45460,
        'validation': 1000
    },
    't0_task_adaptation:anli_based_on_the_previous_passage_r2_score_eval': {
        'test': 3000,
        'train': 136380,
        'validation': 3000
    },
    't0_task_adaptation:anli_based_on_the_previous_passage_r3': {
        'test': 1200,
        'train': 100459,
        'validation': 1200
    },
    't0_task_adaptation:anli_based_on_the_previous_passage_r3_score_eval': {
        'test': 3600,
        'train': 301377,
        'validation': 3600
    },
    't0_task_adaptation:anli_can_we_infer_r1': {
        'test': 1000,
        'train': 16946,
        'validation': 1000
    },
    't0_task_adaptation:anli_can_we_infer_r1_score_eval': {
        'test': 3000,
        'train': 50838,
        'validation': 3000
    },
    't0_task_adaptation:anli_can_we_infer_r2': {
        'test': 1000,
        'train': 45460,
        'validation': 1000
    },
    't0_task_adaptation:anli_can_we_infer_r2_score_eval': {
        'test': 3000,
        'train': 136380,
        'validation': 3000
    },
    't0_task_adaptation:anli_can_we_infer_r3': {
        'test': 1200,
        'train': 100459,
        'validation': 1200
    },
    't0_task_adaptation:anli_can_we_infer_r3_score_eval': {
        'test': 3600,
        'train': 301377,
        'validation': 3600
    },
    't0_task_adaptation:anli_claim_true_false_inconclusive_r1': {
        'test': 1000,
        'train': 16946,
        'validation': 1000
    },
    't0_task_adaptation:anli_claim_true_false_inconclusive_r1_score_eval': {
        'test': 3000,
        'train': 50838,
        'validation': 3000
    },
    't0_task_adaptation:anli_claim_true_false_inconclusive_r2': {
        'test': 1000,
        'train': 45460,
        'validation': 1000
    },
    't0_task_adaptation:anli_claim_true_false_inconclusive_r2_score_eval': {
        'test': 3000,
        'train': 136380,
        'validation': 3000
    },
    't0_task_adaptation:anli_claim_true_false_inconclusive_r3': {
        'test': 1200,
        'train': 100459,
        'validation': 1200
    },
    't0_task_adaptation:anli_claim_true_false_inconclusive_r3_score_eval': {
        'test': 3600,
        'train': 301377,
        'validation': 3600
    },
    't0_task_adaptation:anli_consider_always_sometimes_never_r1': {
        'test': 1000,
        'train': 16946,
        'validation': 1000
    },
    't0_task_adaptation:anli_consider_always_sometimes_never_r1_score_eval': {
        'test': 3000,
        'train': 50838,
        'validation': 3000
    },
    't0_task_adaptation:anli_consider_always_sometimes_never_r2': {
        'test': 1000,
        'train': 45460,
        'validation': 1000
    },
    't0_task_adaptation:anli_consider_always_sometimes_never_r2_score_eval': {
        'test': 3000,
        'train': 136380,
        'validation': 3000
    },
    't0_task_adaptation:anli_consider_always_sometimes_never_r3': {
        'test': 1200,
        'train': 100459,
        'validation': 1200
    },
    't0_task_adaptation:anli_consider_always_sometimes_never_r3_score_eval': {
        'test': 3600,
        'train': 301377,
        'validation': 3600
    },
    't0_task_adaptation:anli_does_it_follow_that_r1': {
        'test': 1000,
        'train': 16946,
        'validation': 1000
    },
    't0_task_adaptation:anli_does_it_follow_that_r1_score_eval': {
        'test': 3000,
        'train': 50838,
        'validation': 3000
    },
    't0_task_adaptation:anli_does_it_follow_that_r2': {
        'test': 1000,
        'train': 45460,
        'validation': 1000
    },
    't0_task_adaptation:anli_does_it_follow_that_r2_score_eval': {
        'test': 3000,
        'train': 136380,
        'validation': 3000
    },
    't0_task_adaptation:anli_does_it_follow_that_r3': {
        'test': 1200,
        'train': 100459,
        'validation': 1200
    },
    't0_task_adaptation:anli_does_it_follow_that_r3_score_eval': {
        'test': 3600,
        'train': 301377,
        'validation': 3600
    },
    't0_task_adaptation:anli_does_this_imply_r1': {
        'test': 1000,
        'train': 16946,
        'validation': 1000
    },
    't0_task_adaptation:anli_does_this_imply_r1_score_eval': {
        'test': 3000,
        'train': 50838,
        'validation': 3000
    },
    't0_task_adaptation:anli_does_this_imply_r2': {
        'test': 1000,
        'train': 45460,
        'validation': 1000
    },
    't0_task_adaptation:anli_does_this_imply_r2_score_eval': {
        'test': 3000,
        'train': 136380,
        'validation': 3000
    },
    't0_task_adaptation:anli_does_this_imply_r3': {
        'test': 1200,
        'train': 100459,
        'validation': 1200
    },
    't0_task_adaptation:anli_does_this_imply_r3_score_eval': {
        'test': 3600,
        'train': 301377,
        'validation': 3600
    },
    't0_task_adaptation:anli_guaranteed_possible_impossible_r1': {
        'test': 1000,
        'train': 16946,
        'validation': 1000
    },
    't0_task_adaptation:anli_guaranteed_possible_impossible_r1_score_eval': {
        'test': 3000,
        'train': 50838,
        'validation': 3000
    },
    't0_task_adaptation:anli_guaranteed_possible_impossible_r2': {
        'test': 1000,
        'train': 45460,
        'validation': 1000
    },
    't0_task_adaptation:anli_guaranteed_possible_impossible_r2_score_eval': {
        'test': 3000,
        'train': 136380,
        'validation': 3000
    },
    't0_task_adaptation:anli_guaranteed_possible_impossible_r3': {
        'test': 1200,
        'train': 100459,
        'validation': 1200
    },
    't0_task_adaptation:anli_guaranteed_possible_impossible_r3_score_eval': {
        'test': 3600,
        'train': 301377,
        'validation': 3600
    },
    't0_task_adaptation:anli_guaranteed_true_r1': {
        'test': 1000,
        'train': 16946,
        'validation': 1000
    },
    't0_task_adaptation:anli_guaranteed_true_r1_score_eval': {
        'test': 3000,
        'train': 50838,
        'validation': 3000
    },
    't0_task_adaptation:anli_guaranteed_true_r2': {
        'test': 1000,
        'train': 45460,
        'validation': 1000
    },
    't0_task_adaptation:anli_guaranteed_true_r2_score_eval': {
        'test': 3000,
        'train': 136380,
        'validation': 3000
    },
    't0_task_adaptation:anli_guaranteed_true_r3': {
        'test': 1200,
        'train': 100459,
        'validation': 1200
    },
    't0_task_adaptation:anli_guaranteed_true_r3_score_eval': {
        'test': 3600,
        'train': 301377,
        'validation': 3600
    },
    't0_task_adaptation:anli_justified_in_saying_r1': {
        'test': 1000,
        'train': 16946,
        'validation': 1000
    },
    't0_task_adaptation:anli_justified_in_saying_r1_score_eval': {
        'test': 3000,
        'train': 50838,
        'validation': 3000
    },
    't0_task_adaptation:anli_justified_in_saying_r2': {
        'test': 1000,
        'train': 45460,
        'validation': 1000
    },
    't0_task_adaptation:anli_justified_in_saying_r2_score_eval': {
        'test': 3000,
        'train': 136380,
        'validation': 3000
    },
    't0_task_adaptation:anli_justified_in_saying_r3': {
        'test': 1200,
        'train': 100459,
        'validation': 1200
    },
    't0_task_adaptation:anli_justified_in_saying_r3_score_eval': {
        'test': 3600,
        'train': 301377,
        'validation': 3600
    },
    't0_task_adaptation:anli_must_be_true_r1': {
        'test': 1000,
        'train': 16946,
        'validation': 1000
    },
    't0_task_adaptation:anli_must_be_true_r1_score_eval': {
        'test': 3000,
        'train': 50838,
        'validation': 3000
    },
    't0_task_adaptation:anli_must_be_true_r2': {
        'test': 1000,
        'train': 45460,
        'validation': 1000
    },
    't0_task_adaptation:anli_must_be_true_r2_score_eval': {
        'test': 3000,
        'train': 136380,
        'validation': 3000
    },
    't0_task_adaptation:anli_must_be_true_r3': {
        'test': 1200,
        'train': 100459,
        'validation': 1200
    },
    't0_task_adaptation:anli_must_be_true_r3_score_eval': {
        'test': 3600,
        'train': 301377,
        'validation': 3600
    },
    't0_task_adaptation:anli_should_assume_r1': {
        'test': 1000,
        'train': 16946,
        'validation': 1000
    },
    't0_task_adaptation:anli_should_assume_r1_score_eval': {
        'test': 3000,
        'train': 50838,
        'validation': 3000
    },
    't0_task_adaptation:anli_should_assume_r2': {
        'test': 1000,
        'train': 45460,
        'validation': 1000
    },
    't0_task_adaptation:anli_should_assume_r2_score_eval': {
        'test': 3000,
        'train': 136380,
        'validation': 3000
    },
    't0_task_adaptation:anli_should_assume_r3': {
        'test': 1200,
        'train': 100459,
        'validation': 1200
    },
    't0_task_adaptation:anli_should_assume_r3_score_eval': {
        'test': 3600,
        'train': 301377,
        'validation': 3600
    },
    't0_task_adaptation:anli_take_the_following_as_truth_r1': {
        'test': 1000,
        'train': 16946,
        'validation': 1000
    },
    't0_task_adaptation:anli_take_the_following_as_truth_r1_score_eval': {
        'test': 3000,
        'train': 50838,
        'validation': 3000
    },
    't0_task_adaptation:anli_take_the_following_as_truth_r2': {
        'test': 1000,
        'train': 45460,
        'validation': 1000
    },
    't0_task_adaptation:anli_take_the_following_as_truth_r2_score_eval': {
        'test': 3000,
        'train': 136380,
        'validation': 3000
    },
    't0_task_adaptation:anli_take_the_following_as_truth_r3': {
        'test': 1200,
        'train': 100459,
        'validation': 1200
    },
    't0_task_adaptation:anli_take_the_following_as_truth_r3_score_eval': {
        'test': 3600,
        'train': 301377,
        'validation': 3600
    },
    't0_task_adaptation:app_reviews_categorize_rating_using_review': {
        'train': 288065
    },
    't0_task_adaptation:app_reviews_convert_to_rating': {
        'train': 288065
    },
    't0_task_adaptation:app_reviews_convert_to_star_rating': {
        'train': 288065
    },
    't0_task_adaptation:app_reviews_generate_review': {
        'train': 288065
    },
    't0_task_adaptation:cnn_dailymail_3.0.0_2_or_3_sentences': {
        'test': 11490,
        'train': 287113,
        'validation': 13368
    },
    't0_task_adaptation:cnn_dailymail_3.0.0_generate_story': {
        'test': 11490,
        'train': 287113,
        'validation': 13368
    },
    't0_task_adaptation:cnn_dailymail_3.0.0_news_card_view': {
        'test': 11490,
        'train': 287113,
        'validation': 13368
    },
    't0_task_adaptation:cnn_dailymail_3.0.0_news_stock': {
        'test': 11490,
        'train': 287113,
        'validation': 13368
    },
    't0_task_adaptation:cnn_dailymail_3.0.0_news_summary': {
        'test': 11490,
        'train': 287113,
        'validation': 13368
    },
    't0_task_adaptation:cnn_dailymail_3.0.0_spice_up_story': {
        'test': 11490,
        'train': 287113,
        'validation': 13368
    },
    't0_task_adaptation:cnn_dailymail_3.0.0_sum_in_brief': {
        'test': 11490,
        'train': 287113,
        'validation': 13368
    },
    't0_task_adaptation:cnn_dailymail_3.0.0_tldr_summary': {
        'test': 11490,
        'train': 287113,
        'validation': 13368
    },
    't0_task_adaptation:cnn_dailymail_3.0.0_write_an_outline': {
        'test': 11490,
        'train': 287113,
        'validation': 13368
    },
    't0_task_adaptation:common_gen_Example_prompt': {
        'test': 1497,
        'train': 67389,
        'validation': 4018
    },
    't0_task_adaptation:common_gen_Given_concepts_type_1': {
        'test': 1497,
        'train': 67389,
        'validation': 4018
    },
    't0_task_adaptation:common_gen_Given_concepts_type_2': {
        'test': 1497,
        'train': 67389,
        'validation': 4018
    },
    't0_task_adaptation:common_gen_Put_together': {
        'test': 1497,
        'train': 67389,
        'validation': 4018
    },
    't0_task_adaptation:common_gen_choice_in_concept_centric_sentence_generation':
        {
            'test': 1497,
            'train': 67389,
            'validation': 4018
        },
    't0_task_adaptation:common_gen_random_task_template_prompt': {
        'test': 1497,
        'train': 67389,
        'validation': 4018
    },
    't0_task_adaptation:common_gen_sentence_to_concepts': {
        'test': 1497,
        'train': 67389,
        'validation': 4018
    },
    't0_task_adaptation:common_gen_topic_to_sentence': {
        'test': 1497,
        'train': 67389,
        'validation': 4018
    },
    't0_task_adaptation:common_gen_topics_from_the_sentence': {
        'test': 1497,
        'train': 67389,
        'validation': 4018
    },
    't0_task_adaptation:cos_e_v1.11_aligned_with_common_sense': {
        'train': 9741,
        'validation': 1221
    },
    't0_task_adaptation:cos_e_v1.11_description_question_option_id': {
        'train': 9741,
        'validation': 1221
    },
    't0_task_adaptation:cos_e_v1.11_description_question_option_text': {
        'train': 9741,
        'validation': 1221
    },
    't0_task_adaptation:cos_e_v1.11_explain_why_human': {
        'train': 9741,
        'validation': 1221
    },
    't0_task_adaptation:cos_e_v1.11_generate_explanation_given_text': {
        'train': 9741,
        'validation': 1221
    },
    't0_task_adaptation:cos_e_v1.11_i_think': {
        'train': 9741,
        'validation': 1221
    },
    't0_task_adaptation:cos_e_v1.11_question_description_option_id': {
        'train': 9741,
        'validation': 1221
    },
    't0_task_adaptation:cos_e_v1.11_question_description_option_text': {
        'train': 9741,
        'validation': 1221
    },
    't0_task_adaptation:cos_e_v1.11_question_option_description_id': {
        'train': 9741,
        'validation': 1221
    },
    't0_task_adaptation:cos_e_v1.11_question_option_description_text': {
        'train': 9741,
        'validation': 1221
    },
    't0_task_adaptation:cos_e_v1.11_rationale': {
        'train': 9741,
        'validation': 1221
    },
    't0_task_adaptation:cosmos_qa_context_answer_to_question': {
        'test': 6963,
        'train': 25262,
        'validation': 2985
    },
    't0_task_adaptation:cosmos_qa_context_description_question_answer_id': {
        'test': 6963,
        'train': 25262,
        'validation': 2985
    },
    't0_task_adaptation:cosmos_qa_context_description_question_answer_text': {
        'test': 6963,
        'train': 25262,
        'validation': 2985
    },
    't0_task_adaptation:cosmos_qa_context_description_question_text': {
        'test': 6963,
        'train': 25262,
        'validation': 2985
    },
    't0_task_adaptation:cosmos_qa_context_question_description_answer_id': {
        'test': 6963,
        'train': 25262,
        'validation': 2985
    },
    't0_task_adaptation:cosmos_qa_context_question_description_answer_text': {
        'test': 6963,
        'train': 25262,
        'validation': 2985
    },
    't0_task_adaptation:cosmos_qa_context_question_description_text': {
        'test': 6963,
        'train': 25262,
        'validation': 2985
    },
    't0_task_adaptation:cosmos_qa_description_context_question_answer_id': {
        'test': 6963,
        'train': 25262,
        'validation': 2985
    },
    't0_task_adaptation:cosmos_qa_description_context_question_answer_text': {
        'test': 6963,
        'train': 25262,
        'validation': 2985
    },
    't0_task_adaptation:cosmos_qa_description_context_question_text': {
        'test': 6963,
        'train': 25262,
        'validation': 2985
    },
    't0_task_adaptation:cosmos_qa_no_prompt_id': {
        'test': 6963,
        'train': 25262,
        'validation': 2985
    },
    't0_task_adaptation:cosmos_qa_no_prompt_text': {
        'test': 6963,
        'train': 25262,
        'validation': 2985
    },
    't0_task_adaptation:cosmos_qa_only_question_answer': {
        'test': 6963,
        'train': 25262,
        'validation': 2985
    },
    't0_task_adaptation:dbpedia_14_given_a_choice_of_categories_': {
        'test': 70000,
        'train': 560000
    },
    't0_task_adaptation:dbpedia_14_given_a_list_of_category_what_does_the_title_belong_to':
        {
            'test': 70000,
            'train': 560000
        },
    't0_task_adaptation:dbpedia_14_given_list_what_category_does_the_paragraph_belong_to':
        {
            'test': 70000,
            'train': 560000
        },
    't0_task_adaptation:dbpedia_14_pick_one_category_for_the_following_text': {
        'test': 70000,
        'train': 560000
    },
    't0_task_adaptation:dream_answer_to_dialogue': {
        'test': 2041,
        'train': 6116,
        'validation': 2040
    },
    't0_task_adaptation:dream_baseline': {
        'test': 2041,
        'train': 6116,
        'validation': 2040
    },
    't0_task_adaptation:dream_generate_first_utterance': {
        'test': 2041,
        'train': 6116,
        'validation': 2040
    },
    't0_task_adaptation:dream_generate_last_utterance': {
        'test': 2041,
        'train': 6116,
        'validation': 2040
    },
    't0_task_adaptation:dream_read_the_following_conversation_and_answer_the_question':
        {
            'test': 2041,
            'train': 6116,
            'validation': 2040
        },
    't0_task_adaptation:duorc_ParaphraseRC_answer_question': {
        'test': 15857,
        'train': 69524,
        'validation': 15591
    },
    't0_task_adaptation:duorc_ParaphraseRC_build_story_around_qa': {
        'test': 13449,
        'train': 58752,
        'validation': 13111
    },
    't0_task_adaptation:duorc_ParaphraseRC_decide_worth_it': {
        'test': 15857,
        'train': 69524,
        'validation': 15591
    },
    't0_task_adaptation:duorc_ParaphraseRC_extract_answer': {
        'test': 15857,
        'train': 69524,
        'validation': 15591
    },
    't0_task_adaptation:duorc_ParaphraseRC_generate_question': {
        'test': 15857,
        'train': 69524,
        'validation': 15591
    },
    't0_task_adaptation:duorc_ParaphraseRC_generate_question_by_answer': {
        'test': 13449,
        'train': 58752,
        'validation': 13111
    },
    't0_task_adaptation:duorc_ParaphraseRC_movie_director': {
        'test': 15857,
        'train': 69524,
        'validation': 15591
    },
    't0_task_adaptation:duorc_ParaphraseRC_question_answering': {
        'test': 15857,
        'train': 69524,
        'validation': 15591
    },
    't0_task_adaptation:duorc_ParaphraseRC_title_generation': {
        'test': 15857,
        'train': 69524,
        'validation': 15591
    },
    't0_task_adaptation:duorc_SelfRC_answer_question': {
        'test': 12559,
        'train': 60721,
        'validation': 12961
    },
    't0_task_adaptation:duorc_SelfRC_build_story_around_qa': {
        'test': 12415,
        'train': 60094,
        'validation': 12845
    },
    't0_task_adaptation:duorc_SelfRC_decide_worth_it': {
        'test': 12559,
        'train': 60721,
        'validation': 12961
    },
    't0_task_adaptation:duorc_SelfRC_extract_answer': {
        'test': 12559,
        'train': 60721,
        'validation': 12961
    },
    't0_task_adaptation:duorc_SelfRC_generate_question': {
        'test': 12559,
        'train': 60721,
        'validation': 12961
    },
    't0_task_adaptation:duorc_SelfRC_generate_question_by_answer': {
        'test': 12415,
        'train': 60094,
        'validation': 12845
    },
    't0_task_adaptation:duorc_SelfRC_movie_director': {
        'test': 12559,
        'train': 60721,
        'validation': 12961
    },
    't0_task_adaptation:duorc_SelfRC_question_answering': {
        'test': 12559,
        'train': 60721,
        'validation': 12961
    },
    't0_task_adaptation:duorc_SelfRC_title_generation': {
        'test': 12559,
        'train': 60721,
        'validation': 12961
    },
    't0_task_adaptation:gigaword_TLDR': {
        'test': 1951,
        'train': 3803957,
        'validation': 189651
    },
    't0_task_adaptation:gigaword_first_sentence_title': {
        'test': 1951,
        'train': 3803957,
        'validation': 189651
    },
    't0_task_adaptation:gigaword_generate_summary_for_this': {
        'test': 1951,
        'train': 3803957,
        'validation': 189651
    },
    't0_task_adaptation:gigaword_in_a_nutshell': {
        'test': 1951,
        'train': 3803957,
        'validation': 189651
    },
    't0_task_adaptation:gigaword_make_a_title': {
        'test': 1951,
        'train': 3803957,
        'validation': 189651
    },
    't0_task_adaptation:gigaword_reverse_writing': {
        'test': 1951,
        'train': 3803957,
        'validation': 189651
    },
    't0_task_adaptation:gigaword_write_a_title_for_this_sentence': {
        'test': 1951,
        'train': 3803957,
        'validation': 189651
    },
    't0_task_adaptation:gigaword_write_an_article': {
        'test': 1951,
        'train': 3803957,
        'validation': 189651
    },
    't0_task_adaptation:gigaword_write_its_sentence': {
        'test': 1951,
        'train': 3803957,
        'validation': 189651
    },
    't0_task_adaptation:glue_mrpc_equivalent': {
        'test': 1725,
        'train': 3668,
        'validation': 408
    },
    't0_task_adaptation:glue_mrpc_generate_paraphrase': {
        'test': 1147,
        'train': 2474,
        'validation': 279
    },
    't0_task_adaptation:glue_mrpc_generate_sentence': {
        'test': 1147,
        'train': 2474,
        'validation': 279
    },
    't0_task_adaptation:glue_mrpc_paraphrase': {
        'test': 1725,
        'train': 3668,
        'validation': 408
    },
    't0_task_adaptation:glue_mrpc_replace': {
        'test': 1725,
        'train': 3668,
        'validation': 408
    },
    't0_task_adaptation:glue_mrpc_same_thing': {
        'test': 1725,
        'train': 3668,
        'validation': 408
    },
    't0_task_adaptation:glue_mrpc_want_to_know': {
        'test': 1725,
        'train': 3668,
        'validation': 408
    },
    't0_task_adaptation:glue_qqp_answer': {
        'test': 390965,
        'train': 363846,
        'validation': 40430
    },
    't0_task_adaptation:glue_qqp_duplicate': {
        'test': 390965,
        'train': 363846,
        'validation': 40430
    },
    't0_task_adaptation:glue_qqp_duplicate_or_not': {
        'test': 390965,
        'train': 363846,
        'validation': 40430
    },
    't0_task_adaptation:glue_qqp_meaning': {
        'test': 390965,
        'train': 363846,
        'validation': 40430
    },
    't0_task_adaptation:glue_qqp_quora': {
        'test': 390965,
        'train': 363846,
        'validation': 40430
    },
    't0_task_adaptation:glue_qqp_same_thing': {
        'test': 390965,
        'train': 363846,
        'validation': 40430
    },
    't0_task_adaptation:hellaswag_Appropriate_continuation_Yes_or_No': {
        'test': 10003,
        'train': 39905,
        'validation': 10042
    },
    't0_task_adaptation:hellaswag_Open_ended_completion': {
        'test': 10003,
        'train': 39905,
        'validation': 10042
    },
    't0_task_adaptation:hellaswag_Open_ended_start': {
        'test': 10003,
        'train': 39905,
        'validation': 10042
    },
    't0_task_adaptation:hellaswag_Predict_ending_with_hint': {
        'test': 10003,
        'train': 39905,
        'validation': 10042
    },
    't0_task_adaptation:hellaswag_Predict_ending_with_hint_score_eval': {
        'test': 40012,
        'train': 159620,
        'validation': 40168
    },
    't0_task_adaptation:hellaswag_Randomized_prompts_template': {
        'test': 10003,
        'train': 39905,
        'validation': 10042
    },
    't0_task_adaptation:hellaswag_Randomized_prompts_template_score_eval': {
        'test': 40012,
        'train': 159620,
        'validation': 40168
    },
    't0_task_adaptation:hellaswag_Reversed_appropriate_continuation_Yes_or_No':
        {
            'test': 10003,
            'train': 39905,
            'validation': 10042
        },
    't0_task_adaptation:hellaswag_Topic_of_the_context': {
        'test': 10003,
        'train': 39905,
        'validation': 10042
    },
    't0_task_adaptation:hellaswag_Topic_without_the_ending_answer': {
        'test': 10003,
        'train': 39905,
        'validation': 10042
    },
    't0_task_adaptation:hellaswag_complete_first_then': {
        'test': 10003,
        'train': 39905,
        'validation': 10042
    },
    't0_task_adaptation:hellaswag_complete_first_then_score_eval': {
        'test': 40012,
        'train': 159620,
        'validation': 40168
    },
    't0_task_adaptation:hellaswag_how_ends': {
        'test': 10003,
        'train': 39905,
        'validation': 10042
    },
    't0_task_adaptation:hellaswag_if_begins_how_continues': {
        'test': 10003,
        'train': 39905,
        'validation': 10042
    },
    't0_task_adaptation:hellaswag_if_begins_how_continues_score_eval': {
        'test': 40012,
        'train': 159620,
        'validation': 40168
    },
    't0_task_adaptation:imdb_Movie_Expressed_Sentiment': {
        'test': 25000,
        'train': 25000,
        'unsupervised': 50000
    },
    't0_task_adaptation:imdb_Movie_Expressed_Sentiment_2': {
        'test': 25000,
        'train': 25000,
        'unsupervised': 50000
    },
    't0_task_adaptation:imdb_Negation_template_for_positive_and_negative': {
        'test': 25000,
        'train': 25000,
        'unsupervised': 50000
    },
    't0_task_adaptation:imdb_Reviewer_Enjoyment': {
        'test': 25000,
        'train': 25000,
        'unsupervised': 50000
    },
    't0_task_adaptation:imdb_Reviewer_Enjoyment_Yes_No': {
        'test': 25000,
        'train': 25000,
        'unsupervised': 50000
    },
    't0_task_adaptation:imdb_Reviewer_Expressed_Sentiment': {
        'test': 25000,
        'train': 25000,
        'unsupervised': 50000
    },
    't0_task_adaptation:imdb_Reviewer_Opinion_bad_good_choices': {
        'test': 25000,
        'train': 25000,
        'unsupervised': 50000
    },
    't0_task_adaptation:imdb_Reviewer_Sentiment_Feeling': {
        'test': 25000,
        'train': 25000,
        'unsupervised': 50000
    },
    't0_task_adaptation:imdb_Sentiment_with_choices_': {
        'test': 25000,
        'train': 25000,
        'unsupervised': 50000
    },
    't0_task_adaptation:imdb_Text_Expressed_Sentiment': {
        'test': 25000,
        'train': 25000,
        'unsupervised': 50000
    },
    't0_task_adaptation:imdb_Writer_Expressed_Sentiment': {
        'test': 25000,
        'train': 25000,
        'unsupervised': 50000
    },
    't0_task_adaptation:kilt_tasks_hotpotqa_combining_facts': {
        'train': 88869,
        'validation': 5600
    },
    't0_task_adaptation:kilt_tasks_hotpotqa_complex_question': {
        'train': 88869,
        'validation': 5600
    },
    't0_task_adaptation:kilt_tasks_hotpotqa_final_exam': {
        'train': 88869,
        'validation': 5600
    },
    't0_task_adaptation:kilt_tasks_hotpotqa_formulate': {
        'train': 88869,
        'validation': 5600
    },
    't0_task_adaptation:kilt_tasks_hotpotqa_straighforward_qa': {
        'train': 88869,
        'validation': 5600
    },
    't0_task_adaptation:multi_news_distill': {
        'test': 5622,
        'train': 44972,
        'validation': 5622
    },
    't0_task_adaptation:multi_news_expand_reverse_task_': {
        'test': 5622,
        'train': 44972,
        'validation': 5622
    },
    't0_task_adaptation:multi_news_summarize': {
        'test': 5622,
        'train': 44972,
        'validation': 5622
    },
    't0_task_adaptation:multi_news_summary_scenario': {
        'test': 5622,
        'train': 44972,
        'validation': 5622
    },
    't0_task_adaptation:multi_news_synthesize': {
        'test': 5622,
        'train': 44972,
        'validation': 5622
    },
    't0_task_adaptation:multi_news_what_are_the_key_points': {
        'test': 5622,
        'train': 44972,
        'validation': 5622
    },
    't0_task_adaptation:openbookqa_main_choices': {
        'test': 500,
        'train': 4957,
        'validation': 500
    },
    't0_task_adaptation:openbookqa_main_choose_an_answer_with_options': {
        'test': 500,
        'train': 4957,
        'validation': 500
    },
    't0_task_adaptation:openbookqa_main_only_options': {
        'test': 500,
        'train': 4957,
        'validation': 500
    },
    't0_task_adaptation:openbookqa_main_pick_answer_with_options': {
        'test': 500,
        'train': 4957,
        'validation': 500
    },
    't0_task_adaptation:openbookqa_main_pick_using_id': {
        'test': 500,
        'train': 4957,
        'validation': 500
    },
    't0_task_adaptation:openbookqa_main_which_correct': {
        'test': 500,
        'train': 4957,
        'validation': 500
    },
    't0_task_adaptation:openbookqa_main_which_correct_inverse': {
        'test': 500,
        'train': 4957,
        'validation': 500
    },
    't0_task_adaptation:paws_labeled_final_Concatenation': {
        'test': 8000,
        'train': 49401,
        'validation': 8000
    },
    't0_task_adaptation:paws_labeled_final_Concatenation_no_label': {
        'test': 8000,
        'train': 49401,
        'validation': 8000
    },
    't0_task_adaptation:paws_labeled_final_Meaning': {
        'test': 8000,
        'train': 49401,
        'validation': 8000
    },
    't0_task_adaptation:paws_labeled_final_Meaning_no_label': {
        'test': 8000,
        'train': 49401,
        'validation': 8000
    },
    't0_task_adaptation:paws_labeled_final_PAWS_ANLI_GPT3': {
        'test': 8000,
        'train': 49401,
        'validation': 8000
    },
    't0_task_adaptation:paws_labeled_final_PAWS_ANLI_GPT3_no_label': {
        'test': 8000,
        'train': 49401,
        'validation': 8000
    },
    't0_task_adaptation:paws_labeled_final_Rewrite': {
        'test': 8000,
        'train': 49401,
        'validation': 8000
    },
    't0_task_adaptation:paws_labeled_final_Rewrite_no_label': {
        'test': 8000,
        'train': 49401,
        'validation': 8000
    },
    't0_task_adaptation:paws_labeled_final_context_question': {
        'test': 8000,
        'train': 49401,
        'validation': 8000
    },
    't0_task_adaptation:paws_labeled_final_context_question_no_label': {
        'test': 8000,
        'train': 49401,
        'validation': 8000
    },
    't0_task_adaptation:paws_labeled_final_paraphrase_task': {
        'test': 3536,
        'train': 21829,
        'validation': 3539
    },
    't0_task_adaptation:paws_labeled_final_task_description_no_label': {
        'test': 8000,
        'train': 49401,
        'validation': 8000
    },
    't0_task_adaptation:piqa_Correct_the_solution': {
        'test': 3084,
        'train': 16113,
        'validation': 1838
    },
    't0_task_adaptation:piqa_Correct_the_solution_if_false_from_sol_1': {
        'test': 3084,
        'train': 16113,
        'validation': 1838
    },
    't0_task_adaptation:piqa_Correct_the_solution_if_false_from_sol_2': {
        'test': 3084,
        'train': 16113,
        'validation': 1838
    },
    't0_task_adaptation:piqa_Does_this_solution_make_sense_sol1': {
        'test': 3084,
        'train': 16113,
        'validation': 1838
    },
    't0_task_adaptation:piqa_Does_this_solution_make_sense_sol2': {
        'test': 3084,
        'train': 16113,
        'validation': 1838
    },
    't0_task_adaptation:piqa_choose_the_most_appropriate_solution': {
        'test': 3084,
        'train': 16113,
        'validation': 1838
    },
    't0_task_adaptation:piqa_finish_sentence_with_correct_choice': {
        'test': 3084,
        'train': 16113,
        'validation': 1838
    },
    't0_task_adaptation:piqa_no_prompt_needed': {
        'test': 3084,
        'train': 16113,
        'validation': 1838
    },
    't0_task_adaptation:piqa_pick_correct_choice_index': {
        'test': 3084,
        'train': 16113,
        'validation': 1838
    },
    't0_task_adaptation:piqa_pick_correct_choice_with_choice_given_before_goal':
        {
            'test': 3084,
            'train': 16113,
            'validation': 1838
        },
    't0_task_adaptation:piqa_what_is_the_correct_ending': {
        'test': 3084,
        'train': 16113,
        'validation': 1838
    },
    't0_task_adaptation:qasc_is_correct_1': {
        'test': 920,
        'train': 8134,
        'validation': 926
    },
    't0_task_adaptation:qasc_is_correct_2': {
        'test': 920,
        'train': 8134,
        'validation': 926
    },
    't0_task_adaptation:qasc_qa_with_combined_facts_1': {
        'test': 920,
        'train': 8134,
        'validation': 926
    },
    't0_task_adaptation:qasc_qa_with_separated_facts_1': {
        'test': 920,
        'train': 8134,
        'validation': 926
    },
    't0_task_adaptation:qasc_qa_with_separated_facts_2': {
        'test': 920,
        'train': 8134,
        'validation': 926
    },
    't0_task_adaptation:qasc_qa_with_separated_facts_3': {
        'test': 920,
        'train': 8134,
        'validation': 926
    },
    't0_task_adaptation:qasc_qa_with_separated_facts_4': {
        'test': 920,
        'train': 8134,
        'validation': 926
    },
    't0_task_adaptation:qasc_qa_with_separated_facts_5': {
        'test': 920,
        'train': 8134,
        'validation': 926
    },
    't0_task_adaptation:quail_context_description_question_answer_id': {
        'challenge': 556,
        'train': 10246,
        'validation': 2164
    },
    't0_task_adaptation:quail_context_description_question_answer_text': {
        'challenge': 556,
        'train': 10246,
        'validation': 2164
    },
    't0_task_adaptation:quail_context_description_question_text': {
        'challenge': 556,
        'train': 10246,
        'validation': 2164
    },
    't0_task_adaptation:quail_context_question_answer_description_id': {
        'challenge': 556,
        'train': 10246,
        'validation': 2164
    },
    't0_task_adaptation:quail_context_question_answer_description_text': {
        'challenge': 556,
        'train': 10246,
        'validation': 2164
    },
    't0_task_adaptation:quail_context_question_description_answer_id': {
        'challenge': 556,
        'train': 10246,
        'validation': 2164
    },
    't0_task_adaptation:quail_context_question_description_answer_text': {
        'challenge': 556,
        'train': 10246,
        'validation': 2164
    },
    't0_task_adaptation:quail_context_question_description_text': {
        'challenge': 556,
        'train': 10246,
        'validation': 2164
    },
    't0_task_adaptation:quail_description_context_question_answer_id': {
        'challenge': 556,
        'train': 10246,
        'validation': 2164
    },
    't0_task_adaptation:quail_description_context_question_answer_text': {
        'challenge': 556,
        'train': 10246,
        'validation': 2164
    },
    't0_task_adaptation:quail_description_context_question_text': {
        'challenge': 556,
        'train': 10246,
        'validation': 2164
    },
    't0_task_adaptation:quail_no_prompt_id': {
        'challenge': 556,
        'train': 10246,
        'validation': 2164
    },
    't0_task_adaptation:quail_no_prompt_text': {
        'challenge': 556,
        'train': 10246,
        'validation': 2164
    },
    't0_task_adaptation:quarel_choose_between': {
        'test': 552,
        'train': 1941,
        'validation': 278
    },
    't0_task_adaptation:quarel_do_not_use': {
        'test': 552,
        'train': 1941,
        'validation': 278
    },
    't0_task_adaptation:quarel_heres_a_story': {
        'test': 552,
        'train': 1941,
        'validation': 278
    },
    't0_task_adaptation:quarel_logic_test': {
        'test': 552,
        'train': 1941,
        'validation': 278
    },
    't0_task_adaptation:quarel_testing_students': {
        'test': 552,
        'train': 1941,
        'validation': 278
    },
    't0_task_adaptation:quartz_answer_question_based_on': {
        'test': 784,
        'train': 2696,
        'validation': 384
    },
    't0_task_adaptation:quartz_answer_question_below': {
        'test': 784,
        'train': 2696,
        'validation': 384
    },
    't0_task_adaptation:quartz_given_the_fact_answer_the_q': {
        'test': 784,
        'train': 2696,
        'validation': 384
    },
    't0_task_adaptation:quartz_having_read_above_passage': {
        'test': 784,
        'train': 2696,
        'validation': 384
    },
    't0_task_adaptation:quartz_paragraph_question_plain_concat': {
        'test': 784,
        'train': 2696,
        'validation': 384
    },
    't0_task_adaptation:quartz_read_passage_below_choose': {
        'test': 784,
        'train': 2696,
        'validation': 384
    },
    't0_task_adaptation:quartz_use_info_from_paragraph_question': {
        'test': 784,
        'train': 2696,
        'validation': 384
    },
    't0_task_adaptation:quartz_use_info_from_question_paragraph': {
        'test': 784,
        'train': 2696,
        'validation': 384
    },
    't0_task_adaptation:quoref_Answer_Friend_Question': {
        'train': 19399,
        'validation': 2418
    },
    't0_task_adaptation:quoref_Answer_Question_Given_Context': {
        'train': 19399,
        'validation': 2418
    },
    't0_task_adaptation:quoref_Answer_Test': {
        'train': 19399,
        'validation': 2418
    },
    't0_task_adaptation:quoref_Context_Contains_Answer': {
        'train': 19399,
        'validation': 2418
    },
    't0_task_adaptation:quoref_Find_Answer': {
        'train': 19399,
        'validation': 2418
    },
    't0_task_adaptation:quoref_Found_Context_Online': {
        'train': 19399,
        'validation': 2418
    },
    't0_task_adaptation:quoref_Given_Context_Answer_Question': {
        'train': 19399,
        'validation': 2418
    },
    't0_task_adaptation:quoref_Guess_Answer': {
        'train': 19399,
        'validation': 2418
    },
    't0_task_adaptation:quoref_Guess_Title_For_Context': {
        'train': 19399,
        'validation': 2418
    },
    't0_task_adaptation:quoref_Read_And_Extract_': {
        'train': 19399,
        'validation': 2418
    },
    't0_task_adaptation:quoref_What_Is_The_Answer': {
        'train': 19399,
        'validation': 2418
    },
    't0_task_adaptation:race_high_Is_this_the_right_answer': {
        'test': 3498,
        'train': 62445,
        'validation': 3451
    },
    't0_task_adaptation:race_high_Read_the_article_and_answer_the_question_no_option_':
        {
            'test': 3498,
            'train': 62445,
            'validation': 3451
        },
    't0_task_adaptation:race_high_Select_the_best_answer': {
        'test': 3498,
        'train': 62445,
        'validation': 3451
    },
    't0_task_adaptation:race_high_Select_the_best_answer_generate_span_': {
        'test': 3498,
        'train': 62445,
        'validation': 3451
    },
    't0_task_adaptation:race_high_Select_the_best_answer_no_instructions_': {
        'test': 3498,
        'train': 62445,
        'validation': 3451
    },
    't0_task_adaptation:race_high_Taking_a_test': {
        'test': 3498,
        'train': 62445,
        'validation': 3451
    },
    't0_task_adaptation:race_high_Write_a_multi_choice_question_for_the_following_article':
        {
            'test': 3498,
            'train': 62445,
            'validation': 3451
        },
    't0_task_adaptation:race_high_Write_a_multi_choice_question_options_given_':
        {
            'test': 3498,
            'train': 62445,
            'validation': 3451
        },
    't0_task_adaptation:race_middle_Is_this_the_right_answer': {
        'test': 1436,
        'train': 25421,
        'validation': 1436
    },
    't0_task_adaptation:race_middle_Read_the_article_and_answer_the_question_no_option_':
        {
            'test': 1436,
            'train': 25421,
            'validation': 1436
        },
    't0_task_adaptation:race_middle_Select_the_best_answer': {
        'test': 1436,
        'train': 25421,
        'validation': 1436
    },
    't0_task_adaptation:race_middle_Select_the_best_answer_generate_span_': {
        'test': 1436,
        'train': 25421,
        'validation': 1436
    },
    't0_task_adaptation:race_middle_Select_the_best_answer_no_instructions_': {
        'test': 1436,
        'train': 25421,
        'validation': 1436
    },
    't0_task_adaptation:race_middle_Taking_a_test': {
        'test': 1436,
        'train': 25421,
        'validation': 1436
    },
    't0_task_adaptation:race_middle_Write_a_multi_choice_question_for_the_following_article':
        {
            'test': 1436,
            'train': 25421,
            'validation': 1436
        },
    't0_task_adaptation:race_middle_Write_a_multi_choice_question_options_given_':
        {
            'test': 1436,
            'train': 25421,
            'validation': 1436
        },
    't0_task_adaptation:ropes_background_new_situation_answer': {
        'train': 10924,
        'validation': 1688
    },
    't0_task_adaptation:ropes_background_situation_middle': {
        'train': 10924,
        'validation': 1688
    },
    't0_task_adaptation:ropes_given_background_situation': {
        'train': 10924,
        'validation': 1688
    },
    't0_task_adaptation:ropes_new_situation_background_answer': {
        'train': 10924,
        'validation': 1688
    },
    't0_task_adaptation:ropes_plain_background_situation': {
        'train': 10924,
        'validation': 1688
    },
    't0_task_adaptation:ropes_plain_bottom_hint': {
        'train': 10924,
        'validation': 1688
    },
    't0_task_adaptation:ropes_plain_no_background': {
        'train': 10924,
        'validation': 1688
    },
    't0_task_adaptation:ropes_prompt_beginning': {
        'train': 10924,
        'validation': 1688
    },
    't0_task_adaptation:ropes_prompt_bottom_hint_beginning': {
        'train': 10924,
        'validation': 1688
    },
    't0_task_adaptation:ropes_prompt_bottom_no_hint': {
        'train': 10924,
        'validation': 1688
    },
    't0_task_adaptation:ropes_prompt_mix': {
        'train': 10924,
        'validation': 1688
    },
    't0_task_adaptation:ropes_read_background_situation': {
        'train': 10924,
        'validation': 1688
    },
    't0_task_adaptation:rotten_tomatoes_Movie_Expressed_Sentiment': {
        'test': 1066,
        'train': 8530,
        'validation': 1066
    },
    't0_task_adaptation:rotten_tomatoes_Movie_Expressed_Sentiment_2': {
        'test': 1066,
        'train': 8530,
        'validation': 1066
    },
    't0_task_adaptation:rotten_tomatoes_Reviewer_Enjoyment': {
        'test': 1066,
        'train': 8530,
        'validation': 1066
    },
    't0_task_adaptation:rotten_tomatoes_Reviewer_Enjoyment_Yes_No': {
        'test': 1066,
        'train': 8530,
        'validation': 1066
    },
    't0_task_adaptation:rotten_tomatoes_Reviewer_Expressed_Sentiment': {
        'test': 1066,
        'train': 8530,
        'validation': 1066
    },
    't0_task_adaptation:rotten_tomatoes_Reviewer_Opinion_bad_good_choices': {
        'test': 1066,
        'train': 8530,
        'validation': 1066
    },
    't0_task_adaptation:rotten_tomatoes_Reviewer_Sentiment_Feeling': {
        'test': 1066,
        'train': 8530,
        'validation': 1066
    },
    't0_task_adaptation:rotten_tomatoes_Sentiment_with_choices_': {
        'test': 1066,
        'train': 8530,
        'validation': 1066
    },
    't0_task_adaptation:rotten_tomatoes_Text_Expressed_Sentiment': {
        'test': 1066,
        'train': 8530,
        'validation': 1066
    },
    't0_task_adaptation:rotten_tomatoes_Writer_Expressed_Sentiment': {
        'test': 1066,
        'train': 8530,
        'validation': 1066
    },
    't0_task_adaptation:samsum_Generate_a_summary_for_this_dialogue': {
        'test': 819,
        'train': 14732,
        'validation': 818
    },
    't0_task_adaptation:samsum_Given_the_above_dialogue_write_a_summary': {
        'test': 819,
        'train': 14732,
        'validation': 818
    },
    't0_task_adaptation:samsum_Sum_up_the_following_dialogue': {
        'test': 819,
        'train': 14732,
        'validation': 818
    },
    't0_task_adaptation:samsum_Summarize_': {
        'test': 819,
        'train': 14732,
        'validation': 818
    },
    't0_task_adaptation:samsum_Summarize_this_dialogue_': {
        'test': 819,
        'train': 14732,
        'validation': 818
    },
    't0_task_adaptation:samsum_To_sum_up_this_dialog': {
        'test': 819,
        'train': 14732,
        'validation': 818
    },
    't0_task_adaptation:samsum_Write_a_dialogue_that_match_this_summary': {
        'test': 819,
        'train': 14732,
        'validation': 818
    },
    't0_task_adaptation:sciq_Direct_Question': {
        'test': 1000,
        'train': 11679,
        'validation': 1000
    },
    't0_task_adaptation:sciq_Direct_Question_Closed_Book_': {
        'test': 1000,
        'train': 11679,
        'validation': 1000
    },
    't0_task_adaptation:sciq_Multiple_Choice': {
        'test': 1000,
        'train': 11679,
        'validation': 1000
    },
    't0_task_adaptation:sciq_Multiple_Choice_Closed_Book_': {
        'test': 1000,
        'train': 11679,
        'validation': 1000
    },
    't0_task_adaptation:sciq_Multiple_Choice_Question_First': {
        'test': 1000,
        'train': 11679,
        'validation': 1000
    },
    't0_task_adaptation:social_i_qa_Check_if_a_random_answer_is_valid_or_not': {
        'train': 33410,
        'validation': 1954
    },
    't0_task_adaptation:social_i_qa_Generate_answer': {
        'train': 33410,
        'validation': 1954
    },
    't0_task_adaptation:social_i_qa_Generate_the_question_from_the_answer': {
        'train': 33410,
        'validation': 1954
    },
    't0_task_adaptation:social_i_qa_I_was_wondering': {
        'train': 33410,
        'validation': 1954
    },
    't0_task_adaptation:social_i_qa_Show_choices_and_generate_answer': {
        'train': 33410,
        'validation': 1954
    },
    't0_task_adaptation:social_i_qa_Show_choices_and_generate_index': {
        'train': 33410,
        'validation': 1954
    },
    't0_task_adaptation:squad_v2_Jeopardy_with_Context': {
        'train': 86821,
        'validation': 5928
    },
    't0_task_adaptation:squad_v2_Jeopardy_without_Context': {
        'train': 86821,
        'validation': 5928
    },
    't0_task_adaptation:squad_v2_Questions_with_Context': {
        'train': 130319,
        'validation': 11873
    },
    't0_task_adaptation:squad_v2_Questions_with_Context_Without_Prompt_Keywords':
        {
            'train': 130319,
            'validation': 11873
        },
    't0_task_adaptation:squad_v2_Questions_with_Context_Without_Prompt_Keywords_unanswerable':
        {
            'train': 130319,
            'validation': 11873
        },
    't0_task_adaptation:squad_v2_Questions_with_Context_unanswerable': {
        'train': 130319,
        'validation': 11873
    },
    't0_task_adaptation:squad_v2_Topic_Prediction_Context': {
        'train': 130319,
        'validation': 11873
    },
    't0_task_adaptation:squad_v2_Topic_Prediction_Context_with_randomized_prompt_options':
        {
            'train': 130319,
            'validation': 11873
        },
    't0_task_adaptation:squad_v2_Topic_Prediction_Context_with_randomized_prompt_options_placed_in_the_end':
        {
            'train': 130319,
            'validation': 11873
        },
    't0_task_adaptation:squad_v2_Topic_Prediction_Question_and_Answer_Pair': {
        'train': 86821,
        'validation': 5928
    },
    't0_task_adaptation:squad_v2_Trivia': {
        'train': 86821,
        'validation': 5928
    },
    't0_task_adaptation:squad_v2_Unanwerable_question': {
        'train': 130319,
        'validation': 11873
    },
    't0_task_adaptation:super_glue_boolq_GPT_3_Style': {
        'test': 3245,
        'train': 9427,
        'validation': 3270
    },
    't0_task_adaptation:super_glue_boolq_I_wonder_': {
        'test': 3245,
        'train': 9427,
        'validation': 3270
    },
    't0_task_adaptation:super_glue_boolq_after_reading': {
        'test': 3245,
        'train': 9427,
        'validation': 3270
    },
    't0_task_adaptation:super_glue_boolq_based_on_the_following_passage': {
        'test': 3245,
        'train': 9427,
        'validation': 3270
    },
    't0_task_adaptation:super_glue_boolq_based_on_the_previous_passage': {
        'test': 3245,
        'train': 9427,
        'validation': 3270
    },
    't0_task_adaptation:super_glue_boolq_could_you_tell_me_': {
        'test': 3245,
        'train': 9427,
        'validation': 3270
    },
    't0_task_adaptation:super_glue_boolq_exam': {
        'test': 3245,
        'train': 9427,
        'validation': 3270
    },
    't0_task_adaptation:super_glue_boolq_exercise': {
        'test': 3245,
        'train': 9427,
        'validation': 3270
    },
    't0_task_adaptation:super_glue_boolq_valid_binary': {
        'test': 3245,
        'train': 9427,
        'validation': 3270
    },
    't0_task_adaptation:super_glue_boolq_yes_no_question': {
        'test': 3245,
        'train': 9427,
        'validation': 3270
    },
    't0_task_adaptation:super_glue_cb_GPT_3_style': {
        'test': 250,
        'train': 250,
        'validation': 56
    },
    't0_task_adaptation:super_glue_cb_GPT_3_style_score_eval': {
        'test': 750,
        'train': 750,
        'validation': 168
    },
    't0_task_adaptation:super_glue_cb_MNLI_crowdsource': {
        'test': 250,
        'train': 250,
        'validation': 56
    },
    't0_task_adaptation:super_glue_cb_MNLI_crowdsource_score_eval': {
        'test': 750,
        'train': 750,
        'validation': 168
    },
    't0_task_adaptation:super_glue_cb_always_sometimes_never': {
        'test': 250,
        'train': 250,
        'validation': 56
    },
    't0_task_adaptation:super_glue_cb_always_sometimes_never_score_eval': {
        'test': 750,
        'train': 750,
        'validation': 168
    },
    't0_task_adaptation:super_glue_cb_based_on_the_previous_passage': {
        'test': 250,
        'train': 250,
        'validation': 56
    },
    't0_task_adaptation:super_glue_cb_based_on_the_previous_passage_score_eval':
        {
            'test': 750,
            'train': 750,
            'validation': 168
        },
    't0_task_adaptation:super_glue_cb_can_we_infer': {
        'test': 250,
        'train': 250,
        'validation': 56
    },
    't0_task_adaptation:super_glue_cb_can_we_infer_score_eval': {
        'test': 750,
        'train': 750,
        'validation': 168
    },
    't0_task_adaptation:super_glue_cb_claim_true_false_inconclusive': {
        'test': 250,
        'train': 250,
        'validation': 56
    },
    't0_task_adaptation:super_glue_cb_claim_true_false_inconclusive_score_eval':
        {
            'test': 750,
            'train': 750,
            'validation': 168
        },
    't0_task_adaptation:super_glue_cb_consider_always_sometimes_never': {
        'test': 250,
        'train': 250,
        'validation': 56
    },
    't0_task_adaptation:super_glue_cb_consider_always_sometimes_never_score_eval':
        {
            'test': 750,
            'train': 750,
            'validation': 168
        },
    't0_task_adaptation:super_glue_cb_does_it_follow_that': {
        'test': 250,
        'train': 250,
        'validation': 56
    },
    't0_task_adaptation:super_glue_cb_does_it_follow_that_score_eval': {
        'test': 750,
        'train': 750,
        'validation': 168
    },
    't0_task_adaptation:super_glue_cb_does_this_imply': {
        'test': 250,
        'train': 250,
        'validation': 56
    },
    't0_task_adaptation:super_glue_cb_does_this_imply_score_eval': {
        'test': 750,
        'train': 750,
        'validation': 168
    },
    't0_task_adaptation:super_glue_cb_guaranteed_possible_impossible': {
        'test': 250,
        'train': 250,
        'validation': 56
    },
    't0_task_adaptation:super_glue_cb_guaranteed_possible_impossible_score_eval':
        {
            'test': 750,
            'train': 750,
            'validation': 168
        },
    't0_task_adaptation:super_glue_cb_guaranteed_true': {
        'test': 250,
        'train': 250,
        'validation': 56
    },
    't0_task_adaptation:super_glue_cb_guaranteed_true_score_eval': {
        'test': 750,
        'train': 750,
        'validation': 168
    },
    't0_task_adaptation:super_glue_cb_justified_in_saying': {
        'test': 250,
        'train': 250,
        'validation': 56
    },
    't0_task_adaptation:super_glue_cb_justified_in_saying_score_eval': {
        'test': 750,
        'train': 750,
        'validation': 168
    },
    't0_task_adaptation:super_glue_cb_must_be_true': {
        'test': 250,
        'train': 250,
        'validation': 56
    },
    't0_task_adaptation:super_glue_cb_must_be_true_score_eval': {
        'test': 750,
        'train': 750,
        'validation': 168
    },
    't0_task_adaptation:super_glue_cb_should_assume': {
        'test': 250,
        'train': 250,
        'validation': 56
    },
    't0_task_adaptation:super_glue_cb_should_assume_score_eval': {
        'test': 750,
        'train': 750,
        'validation': 168
    },
    't0_task_adaptation:super_glue_cb_take_the_following_as_truth': {
        'test': 250,
        'train': 250,
        'validation': 56
    },
    't0_task_adaptation:super_glue_cb_take_the_following_as_truth_score_eval': {
        'test': 750,
        'train': 750,
        'validation': 168
    },
    't0_task_adaptation:super_glue_copa_C1_or_C2_premise_so_because_': {
        'test': 500,
        'train': 400,
        'validation': 100
    },
    't0_task_adaptation:super_glue_copa_C1_or_C2_premise_so_because__score_eval':
        {
            'test': 1000,
            'train': 800,
            'validation': 200
        },
    't0_task_adaptation:super_glue_copa__As_a_result_C1_or_C2_': {
        'test': 250,
        'train': 202,
        'validation': 48
    },
    't0_task_adaptation:super_glue_copa__As_a_result_C1_or_C2__score_eval': {
        'test': 500,
        'train': 404,
        'validation': 96
    },
    't0_task_adaptation:super_glue_copa__What_could_happen_next_C1_or_C2_': {
        'test': 250,
        'train': 202,
        'validation': 48
    },
    't0_task_adaptation:super_glue_copa__What_could_happen_next_C1_or_C2__score_eval':
        {
            'test': 500,
            'train': 404,
            'validation': 96
        },
    't0_task_adaptation:super_glue_copa__which_may_be_caused_by': {
        'test': 250,
        'train': 198,
        'validation': 52
    },
    't0_task_adaptation:super_glue_copa__which_may_be_caused_by_score_eval': {
        'test': 500,
        'train': 396,
        'validation': 104
    },
    't0_task_adaptation:super_glue_copa__why_C1_or_C2': {
        'test': 250,
        'train': 198,
        'validation': 52
    },
    't0_task_adaptation:super_glue_copa__why_C1_or_C2_score_eval': {
        'test': 500,
        'train': 396,
        'validation': 104
    },
    't0_task_adaptation:super_glue_copa_best_option': {
        'test': 500,
        'train': 400,
        'validation': 100
    },
    't0_task_adaptation:super_glue_copa_best_option_score_eval': {
        'test': 1000,
        'train': 800,
        'validation': 200
    },
    't0_task_adaptation:super_glue_copa_cause_effect': {
        'test': 500,
        'train': 400,
        'validation': 100
    },
    't0_task_adaptation:super_glue_copa_cause_effect_score_eval': {
        'test': 1000,
        'train': 800,
        'validation': 200
    },
    't0_task_adaptation:super_glue_copa_choose': {
        'test': 500,
        'train': 400,
        'validation': 100
    },
    't0_task_adaptation:super_glue_copa_choose_score_eval': {
        'test': 1000,
        'train': 800,
        'validation': 200
    },
    't0_task_adaptation:super_glue_copa_exercise': {
        'test': 500,
        'train': 400,
        'validation': 100
    },
    't0_task_adaptation:super_glue_copa_exercise_score_eval': {
        'test': 1000,
        'train': 800,
        'validation': 200
    },
    't0_task_adaptation:super_glue_copa_i_am_hesitating': {
        'test': 500,
        'train': 400,
        'validation': 100
    },
    't0_task_adaptation:super_glue_copa_i_am_hesitating_score_eval': {
        'test': 1000,
        'train': 800,
        'validation': 200
    },
    't0_task_adaptation:super_glue_copa_more_likely': {
        'test': 500,
        'train': 400,
        'validation': 100
    },
    't0_task_adaptation:super_glue_copa_more_likely_score_eval': {
        'test': 1000,
        'train': 800,
        'validation': 200
    },
    't0_task_adaptation:super_glue_copa_plausible_alternatives': {
        'test': 500,
        'train': 400,
        'validation': 100
    },
    't0_task_adaptation:super_glue_copa_plausible_alternatives_score_eval': {
        'test': 1000,
        'train': 800,
        'validation': 200
    },
    't0_task_adaptation:super_glue_multirc_I_was_going_to_say_': {
        'test': 9693,
        'train': 27243,
        'validation': 4848
    },
    't0_task_adaptation:super_glue_multirc_Would_it_be_good_to_answer_': {
        'test': 9693,
        'train': 27243,
        'validation': 4848
    },
    't0_task_adaptation:super_glue_multirc_confirm': {
        'test': 9693,
        'train': 27243,
        'validation': 4848
    },
    't0_task_adaptation:super_glue_multirc_correct': {
        'test': 9693,
        'train': 27243,
        'validation': 4848
    },
    't0_task_adaptation:super_glue_multirc_decide_valid': {
        'test': 9693,
        'train': 27243,
        'validation': 4848
    },
    't0_task_adaptation:super_glue_multirc_found_this_answer': {
        'test': 9693,
        'train': 27243,
        'validation': 4848
    },
    't0_task_adaptation:super_glue_multirc_grading': {
        'test': 9693,
        'train': 27243,
        'validation': 4848
    },
    't0_task_adaptation:super_glue_multirc_is_a_correct_answer_': {
        'test': 9693,
        'train': 27243,
        'validation': 4848
    },
    't0_task_adaptation:super_glue_multirc_is_the_correct_answer_': {
        'test': 9693,
        'train': 27243,
        'validation': 4848
    },
    't0_task_adaptation:super_glue_multirc_paragraph_question_is_it_': {
        'test': 9693,
        'train': 27243,
        'validation': 4848
    },
    't0_task_adaptation:super_glue_record_Add_sentence_after_after_continuation_choices_':
        {
            'test': 10000,
            'train': 100730,
            'validation': 10000
        },
    't0_task_adaptation:super_glue_record_Add_sentence_after_continuation_choices_':
        {
            'test': 10000,
            'train': 100730,
            'validation': 10000
        },
    't0_task_adaptation:super_glue_record_Can_you_figure_out_': {
        'test': 10000,
        'train': 100730,
        'validation': 10000
    },
    't0_task_adaptation:super_glue_record_GPT_3_style_continuation_choices_': {
        'test': 10000,
        'train': 100730,
        'validation': 10000
    },
    't0_task_adaptation:super_glue_record_GPT_3_style_summary_only_continuation_choices_':
        {
            'test': 10000,
            'train': 100730,
            'validation': 10000
        },
    't0_task_adaptation:super_glue_record_GPT_3_style_with_labels_continuation_choices_':
        {
            'test': 10000,
            'train': 100730,
            'validation': 10000
        },
    't0_task_adaptation:super_glue_record_GPT_3_style_with_labels_without_hyphens_continuation_choices_':
        {
            'test': 10000,
            'train': 100730,
            'validation': 10000
        },
    't0_task_adaptation:super_glue_record_GPT_3_style_without_hyphens_continuation_choices_':
        {
            'test': 10000,
            'train': 100730,
            'validation': 10000
        },
    't0_task_adaptation:super_glue_record_In_the_question_above_the_placeholder_stands_for':
        {
            'test': 10000,
            'train': 100730,
            'validation': 10000
        },
    't0_task_adaptation:super_glue_record_New_highlight_continuation_choices_':
        {
            'test': 10000,
            'train': 100730,
            'validation': 10000
        },
    't0_task_adaptation:super_glue_record_News_article_continuation_choices_': {
        'test': 10000,
        'train': 100730,
        'validation': 10000
    },
    't0_task_adaptation:super_glue_record_Summary_first_continuation_choices_':
        {
            'test': 10000,
            'train': 100730,
            'validation': 10000
        },
    't0_task_adaptation:super_glue_record_What_could_the_placeholder_be_': {
        'test': 10000,
        'train': 100730,
        'validation': 10000
    },
    't0_task_adaptation:super_glue_record_Which_one_is_the_placeholder_': {
        'test': 10000,
        'train': 100730,
        'validation': 10000
    },
    't0_task_adaptation:super_glue_record_choose_between': {
        'test': 10000,
        'train': 100730,
        'validation': 10000
    },
    't0_task_adaptation:super_glue_record_corrupted': {
        'test': 10000,
        'train': 100730,
        'validation': 10000
    },
    't0_task_adaptation:super_glue_record_exercise': {
        'test': 10000,
        'train': 100730,
        'validation': 10000
    },
    't0_task_adaptation:super_glue_record_pick_one_option': {
        'test': 10000,
        'train': 100730,
        'validation': 10000
    },
    't0_task_adaptation:super_glue_record_the_placeholder_refers_to_': {
        'test': 10000,
        'train': 100730,
        'validation': 10000
    },
    't0_task_adaptation:super_glue_record_trying_to_decide': {
        'test': 10000,
        'train': 100730,
        'validation': 10000
    },
    't0_task_adaptation:super_glue_rte_GPT_3_style': {
        'test': 3000,
        'train': 2490,
        'validation': 277
    },
    't0_task_adaptation:super_glue_rte_GPT_3_style_score_eval': {
        'test': 6000,
        'train': 4980,
        'validation': 554
    },
    't0_task_adaptation:super_glue_rte_MNLI_crowdsource': {
        'test': 3000,
        'train': 2490,
        'validation': 277
    },
    't0_task_adaptation:super_glue_rte_MNLI_crowdsource_score_eval': {
        'test': 6000,
        'train': 4980,
        'validation': 554
    },
    't0_task_adaptation:super_glue_rte_based_on_the_previous_passage': {
        'test': 3000,
        'train': 2490,
        'validation': 277
    },
    't0_task_adaptation:super_glue_rte_based_on_the_previous_passage_score_eval':
        {
            'test': 6000,
            'train': 4980,
            'validation': 554
        },
    't0_task_adaptation:super_glue_rte_can_we_infer': {
        'test': 3000,
        'train': 2490,
        'validation': 277
    },
    't0_task_adaptation:super_glue_rte_can_we_infer_score_eval': {
        'test': 6000,
        'train': 4980,
        'validation': 554
    },
    't0_task_adaptation:super_glue_rte_does_it_follow_that': {
        'test': 3000,
        'train': 2490,
        'validation': 277
    },
    't0_task_adaptation:super_glue_rte_does_it_follow_that_score_eval': {
        'test': 6000,
        'train': 4980,
        'validation': 554
    },
    't0_task_adaptation:super_glue_rte_does_this_imply': {
        'test': 3000,
        'train': 2490,
        'validation': 277
    },
    't0_task_adaptation:super_glue_rte_does_this_imply_score_eval': {
        'test': 6000,
        'train': 4980,
        'validation': 554
    },
    't0_task_adaptation:super_glue_rte_guaranteed_true': {
        'test': 3000,
        'train': 2490,
        'validation': 277
    },
    't0_task_adaptation:super_glue_rte_guaranteed_true_score_eval': {
        'test': 6000,
        'train': 4980,
        'validation': 554
    },
    't0_task_adaptation:super_glue_rte_justified_in_saying': {
        'test': 3000,
        'train': 2490,
        'validation': 277
    },
    't0_task_adaptation:super_glue_rte_justified_in_saying_score_eval': {
        'test': 6000,
        'train': 4980,
        'validation': 554
    },
    't0_task_adaptation:super_glue_rte_must_be_true': {
        'test': 3000,
        'train': 2490,
        'validation': 277
    },
    't0_task_adaptation:super_glue_rte_must_be_true_score_eval': {
        'test': 6000,
        'train': 4980,
        'validation': 554
    },
    't0_task_adaptation:super_glue_rte_should_assume': {
        'test': 3000,
        'train': 2490,
        'validation': 277
    },
    't0_task_adaptation:super_glue_rte_should_assume_score_eval': {
        'test': 6000,
        'train': 4980,
        'validation': 554
    },
    't0_task_adaptation:super_glue_wic_GPT_3_prompt': {
        'test': 1400,
        'train': 5428,
        'validation': 638
    },
    't0_task_adaptation:super_glue_wic_GPT_3_prompt_score_eval': {
        'test': 2800,
        'train': 10856,
        'validation': 1276
    },
    't0_task_adaptation:super_glue_wic_GPT_3_prompt_with_label': {
        'test': 1400,
        'train': 5428,
        'validation': 638
    },
    't0_task_adaptation:super_glue_wic_GPT_3_prompt_with_label_score_eval': {
        'test': 2800,
        'train': 10856,
        'validation': 1276
    },
    't0_task_adaptation:super_glue_wic_affirmation_true_or_false': {
        'test': 1400,
        'train': 5428,
        'validation': 638
    },
    't0_task_adaptation:super_glue_wic_affirmation_true_or_false_score_eval': {
        'test': 2800,
        'train': 10856,
        'validation': 1276
    },
    't0_task_adaptation:super_glue_wic_grammar_homework': {
        'test': 1400,
        'train': 5428,
        'validation': 638
    },
    't0_task_adaptation:super_glue_wic_grammar_homework_score_eval': {
        'test': 2800,
        'train': 10856,
        'validation': 1276
    },
    't0_task_adaptation:super_glue_wic_polysemous': {
        'test': 1400,
        'train': 5428,
        'validation': 638
    },
    't0_task_adaptation:super_glue_wic_polysemous_score_eval': {
        'test': 2800,
        'train': 10856,
        'validation': 1276
    },
    't0_task_adaptation:super_glue_wic_question_context': {
        'test': 1400,
        'train': 5428,
        'validation': 638
    },
    't0_task_adaptation:super_glue_wic_question_context_meaning': {
        'test': 1400,
        'train': 5428,
        'validation': 638
    },
    't0_task_adaptation:super_glue_wic_question_context_meaning_score_eval': {
        'test': 2800,
        'train': 10856,
        'validation': 1276
    },
    't0_task_adaptation:super_glue_wic_question_context_meaning_with_label': {
        'test': 1400,
        'train': 5428,
        'validation': 638
    },
    't0_task_adaptation:super_glue_wic_question_context_meaning_with_label_score_eval':
        {
            'test': 2800,
            'train': 10856,
            'validation': 1276
        },
    't0_task_adaptation:super_glue_wic_question_context_score_eval': {
        'test': 2800,
        'train': 10856,
        'validation': 1276
    },
    't0_task_adaptation:super_glue_wic_same_sense': {
        'test': 1400,
        'train': 5428,
        'validation': 638
    },
    't0_task_adaptation:super_glue_wic_same_sense_score_eval': {
        'test': 2800,
        'train': 10856,
        'validation': 1276
    },
    't0_task_adaptation:super_glue_wic_similar_sense': {
        'test': 1400,
        'train': 5428,
        'validation': 638
    },
    't0_task_adaptation:super_glue_wic_similar_sense_score_eval': {
        'test': 2800,
        'train': 10856,
        'validation': 1276
    },
    't0_task_adaptation:super_glue_wsc.fixed_GPT_3_Style': {
        'test': 146,
        'train': 554,
        'validation': 104
    },
    't0_task_adaptation:super_glue_wsc.fixed_GPT_3_Style_score_eval': {
        'test': 292,
        'train': 1108,
        'validation': 208
    },
    't0_task_adaptation:super_glue_wsc.fixed_I_think_they_mean': {
        'test': 146,
        'train': 554,
        'validation': 104
    },
    't0_task_adaptation:super_glue_wsc.fixed_I_think_they_mean_score_eval': {
        'test': 292,
        'train': 1108,
        'validation': 208
    },
    't0_task_adaptation:super_glue_wsc.fixed_Who_or_what_is_are': {
        'test': 146,
        'train': 554,
        'validation': 104
    },
    't0_task_adaptation:super_glue_wsc.fixed_Who_or_what_is_are_score_eval': {
        'test': 292,
        'train': 1108,
        'validation': 208
    },
    't0_task_adaptation:super_glue_wsc.fixed_by_p_they_mean': {
        'test': 146,
        'train': 554,
        'validation': 104
    },
    't0_task_adaptation:super_glue_wsc.fixed_by_p_they_mean_score_eval': {
        'test': 292,
        'train': 1108,
        'validation': 208
    },
    't0_task_adaptation:super_glue_wsc.fixed_does_p_stand_for': {
        'test': 146,
        'train': 554,
        'validation': 104
    },
    't0_task_adaptation:super_glue_wsc.fixed_does_p_stand_for_score_eval': {
        'test': 292,
        'train': 1108,
        'validation': 208
    },
    't0_task_adaptation:super_glue_wsc.fixed_does_the_pronoun_refer_to': {
        'test': 146,
        'train': 554,
        'validation': 104
    },
    't0_task_adaptation:super_glue_wsc.fixed_does_the_pronoun_refer_to_score_eval':
        {
            'test': 292,
            'train': 1108,
            'validation': 208
        },
    't0_task_adaptation:super_glue_wsc.fixed_in_other_words': {
        'test': 146,
        'train': 554,
        'validation': 104
    },
    't0_task_adaptation:super_glue_wsc.fixed_in_other_words_score_eval': {
        'test': 292,
        'train': 1108,
        'validation': 208
    },
    't0_task_adaptation:super_glue_wsc.fixed_p_is_are_r': {
        'test': 146,
        'train': 554,
        'validation': 104
    },
    't0_task_adaptation:super_glue_wsc.fixed_p_is_are_r_score_eval': {
        'test': 292,
        'train': 1108,
        'validation': 208
    },
    't0_task_adaptation:super_glue_wsc.fixed_replaced_with': {
        'test': 146,
        'train': 554,
        'validation': 104
    },
    't0_task_adaptation:super_glue_wsc.fixed_replaced_with_score_eval': {
        'test': 292,
        'train': 1108,
        'validation': 208
    },
    't0_task_adaptation:super_glue_wsc.fixed_the_pronoun_refers_to': {
        'test': 146,
        'train': 554,
        'validation': 104
    },
    't0_task_adaptation:super_glue_wsc.fixed_the_pronoun_refers_to_score_eval':
        {
            'test': 292,
            'train': 1108,
            'validation': 208
        },
    't0_task_adaptation:trec_fine_grained_ABBR': {
        'test': 9,
        'train': 86
    },
    't0_task_adaptation:trec_fine_grained_ABBR_context_first': {
        'test': 9,
        'train': 86
    },
    't0_task_adaptation:trec_fine_grained_DESC': {
        'test': 138,
        'train': 1162
    },
    't0_task_adaptation:trec_fine_grained_DESC_context_first': {
        'test': 138,
        'train': 1162
    },
    't0_task_adaptation:trec_fine_grained_ENTY': {
        'test': 94,
        'train': 1250
    },
    't0_task_adaptation:trec_fine_grained_HUM': {
        'test': 65,
        'train': 1223
    },
    't0_task_adaptation:trec_fine_grained_HUM_context_first': {
        'test': 65,
        'train': 1223
    },
    't0_task_adaptation:trec_fine_grained_LOC': {
        'test': 81,
        'train': 835
    },
    't0_task_adaptation:trec_fine_grained_LOC_context_first': {
        'test': 81,
        'train': 835
    },
    't0_task_adaptation:trec_fine_grained_NUM': {
        'test': 113,
        'train': 896
    },
    't0_task_adaptation:trec_fine_grained_NUM_context_first': {
        'test': 113,
        'train': 896
    },
    't0_task_adaptation:trec_fine_grained_open': {
        'test': 500,
        'train': 5452
    },
    't0_task_adaptation:trec_fine_grained_open_context_first': {
        'test': 500,
        'train': 5452
    },
    't0_task_adaptation:trec_pick_the_best_descriptor': {
        'test': 500,
        'train': 5452
    },
    't0_task_adaptation:trec_trec1': {
        'test': 500,
        'train': 5452
    },
    't0_task_adaptation:trec_trec2': {
        'test': 500,
        'train': 5452
    },
    't0_task_adaptation:trec_what_category_best_describe': {
        'test': 500,
        'train': 5452
    },
    't0_task_adaptation:trec_which_category_best_describes': {
        'test': 500,
        'train': 5452
    },
    't0_task_adaptation:trivia_qa_unfiltered_first_person_context': {
        'test': 10832,
        'train': 87622,
        'validation': 11313
    },
    't0_task_adaptation:trivia_qa_unfiltered_formal_description': {
        'test': 10832,
        'train': 87622,
        'validation': 11313
    },
    't0_task_adaptation:trivia_qa_unfiltered_guess_question': {
        'train': 87622,
        'validation': 11313
    },
    't0_task_adaptation:trivia_qa_unfiltered_question_answer': {
        'test': 10832,
        'train': 87622,
        'validation': 11313
    },
    't0_task_adaptation:trivia_qa_unfiltered_question_with_instruction': {
        'test': 10832,
        'train': 87622,
        'validation': 11313
    },
    't0_task_adaptation:web_questions_get_the_answer': {
        'test': 2032,
        'train': 3778
    },
    't0_task_adaptation:web_questions_potential_correct_answer': {
        'test': 2032,
        'train': 3778
    },
    't0_task_adaptation:web_questions_question_answer': {
        'test': 2032,
        'train': 3778
    },
    't0_task_adaptation:web_questions_short_general_knowledge_q': {
        'test': 2032,
        'train': 3778
    },
    't0_task_adaptation:web_questions_whats_the_answer': {
        'test': 2032,
        'train': 3778
    },
    't0_task_adaptation:wiki_bio_comprehension': {
        'test': 72829,
        'train': 582639,
        'val': 72831
    },
    't0_task_adaptation:wiki_bio_guess_person': {
        'test': 72829,
        'train': 582639,
        'val': 72831
    },
    't0_task_adaptation:wiki_bio_key_content': {
        'test': 72829,
        'train': 582639,
        'val': 72831
    },
    't0_task_adaptation:wiki_bio_what_content': {
        'test': 72829,
        'train': 582639,
        'val': 72831
    },
    't0_task_adaptation:wiki_bio_who': {
        'test': 72829,
        'train': 582639,
        'val': 72831
    },
    't0_task_adaptation:wiki_hop_original_choose_best_object_affirmative_1': {
        'train': 43738,
        'validation': 5129
    },
    't0_task_adaptation:wiki_hop_original_choose_best_object_affirmative_2': {
        'train': 43738,
        'validation': 5129
    },
    't0_task_adaptation:wiki_hop_original_choose_best_object_affirmative_3': {
        'train': 43738,
        'validation': 5129
    },
    't0_task_adaptation:wiki_hop_original_choose_best_object_interrogative_1': {
        'train': 43738,
        'validation': 5129
    },
    't0_task_adaptation:wiki_hop_original_choose_best_object_interrogative_2': {
        'train': 43738,
        'validation': 5129
    },
    't0_task_adaptation:wiki_hop_original_explain_relation': {
        'train': 43738,
        'validation': 5129
    },
    't0_task_adaptation:wiki_hop_original_generate_object': {
        'train': 43738,
        'validation': 5129
    },
    't0_task_adaptation:wiki_hop_original_generate_subject': {
        'train': 43738,
        'validation': 5129
    },
    't0_task_adaptation:wiki_hop_original_generate_subject_and_object': {
        'train': 43738,
        'validation': 5129
    },
    't0_task_adaptation:wiki_qa_Decide_good_answer': {
        'test': 6165,
        'train': 20360,
        'validation': 2733
    },
    't0_task_adaptation:wiki_qa_Direct_Answer_to_Question': {
        'test': 293,
        'train': 1040,
        'validation': 140
    },
    't0_task_adaptation:wiki_qa_Generate_Question_from_Topic': {
        'test': 293,
        'train': 1040,
        'validation': 140
    },
    't0_task_adaptation:wiki_qa_Is_This_True_': {
        'test': 6165,
        'train': 20360,
        'validation': 2733
    },
    't0_task_adaptation:wiki_qa_Jeopardy_style': {
        'test': 293,
        'train': 1040,
        'validation': 140
    },
    't0_task_adaptation:wiki_qa_Topic_Prediction_Answer_Only': {
        'test': 293,
        'train': 1040,
        'validation': 140
    },
    't0_task_adaptation:wiki_qa_Topic_Prediction_Question_Only': {
        'test': 293,
        'train': 1040,
        'validation': 140
    },
    't0_task_adaptation:wiki_qa_Topic_Prediction_Question_and_Answer_Pair': {
        'test': 293,
        'train': 1040,
        'validation': 140
    },
    't0_task_adaptation:wiki_qa_automatic_system': {
        'test': 6165,
        'train': 20360,
        'validation': 2733
    },
    't0_task_adaptation:wiki_qa_exercise': {
        'test': 6165,
        'train': 20360,
        'validation': 2733
    },
    't0_task_adaptation:wiki_qa_found_on_google': {
        'test': 6165,
        'train': 20360,
        'validation': 2733
    },
    't0_task_adaptation:winogrande_winogrande_debiased_Replace': {
        'test': 1767,
        'train': 9248,
        'validation': 1267
    },
    't0_task_adaptation:winogrande_winogrande_debiased_Replace_score_eval': {
        'test': 3534,
        'train': 18496,
        'validation': 2534
    },
    't0_task_adaptation:winogrande_winogrande_debiased_does_underscore_refer_to':
        {
            'test': 1767,
            'train': 9248,
            'validation': 1267
        },
    't0_task_adaptation:winogrande_winogrande_debiased_does_underscore_refer_to_score_eval':
        {
            'test': 3534,
            'train': 18496,
            'validation': 2534
        },
    't0_task_adaptation:winogrande_winogrande_debiased_fill_in_the_blank': {
        'test': 1767,
        'train': 9248,
        'validation': 1267
    },
    't0_task_adaptation:winogrande_winogrande_debiased_fill_in_the_blank_score_eval':
        {
            'test': 3534,
            'train': 18496,
            'validation': 2534
        },
    't0_task_adaptation:winogrande_winogrande_debiased_stand_for': {
        'test': 1767,
        'train': 9248,
        'validation': 1267
    },
    't0_task_adaptation:winogrande_winogrande_debiased_stand_for_score_eval': {
        'test': 3534,
        'train': 18496,
        'validation': 2534
    },
    't0_task_adaptation:winogrande_winogrande_debiased_underscore_refer_to': {
        'test': 1767,
        'train': 9248,
        'validation': 1267
    },
    't0_task_adaptation:winogrande_winogrande_debiased_underscore_refer_to_score_eval':
        {
            'test': 3534,
            'train': 18496,
            'validation': 2534
        },
    't0_task_adaptation:winogrande_winogrande_xl_Replace': {
        'test': 1767,
        'train': 40398,
        'validation': 1267
    },
    't0_task_adaptation:winogrande_winogrande_xl_Replace_score_eval': {
        'test': 3534,
        'train': 80796,
        'validation': 2534
    },
    't0_task_adaptation:winogrande_winogrande_xl_does_underscore_refer_to': {
        'test': 1767,
        'train': 40398,
        'validation': 1267
    },
    't0_task_adaptation:winogrande_winogrande_xl_does_underscore_refer_to_score_eval':
        {
            'test': 3534,
            'train': 80796,
            'validation': 2534
        },
    't0_task_adaptation:winogrande_winogrande_xl_fill_in_the_blank': {
        'test': 1767,
        'train': 40398,
        'validation': 1267
    },
    't0_task_adaptation:winogrande_winogrande_xl_fill_in_the_blank_score_eval':
        {
            'test': 3534,
            'train': 80796,
            'validation': 2534
        },
    't0_task_adaptation:winogrande_winogrande_xl_stand_for': {
        'test': 1767,
        'train': 40398,
        'validation': 1267
    },
    't0_task_adaptation:winogrande_winogrande_xl_stand_for_score_eval': {
        'test': 3534,
        'train': 80796,
        'validation': 2534
    },
    't0_task_adaptation:winogrande_winogrande_xl_underscore_refer_to': {
        'test': 1767,
        'train': 40398,
        'validation': 1267
    },
    't0_task_adaptation:winogrande_winogrande_xl_underscore_refer_to_score_eval':
        {
            'test': 3534,
            'train': 80796,
            'validation': 2534
        },
    't0_task_adaptation:wiqa_does_the_supposed_perturbation_have_an_effect': {
        'test': 3003,
        'train': 29808,
        'validation': 6894
    },
    't0_task_adaptation:wiqa_effect_with_label_answer': {
        'test': 3003,
        'train': 29808,
        'validation': 6894
    },
    't0_task_adaptation:wiqa_effect_with_string_answer': {
        'test': 3003,
        'train': 29808,
        'validation': 6894
    },
    't0_task_adaptation:wiqa_what_is_the_final_step_of_the_following_process': {
        'test': 3003,
        'train': 29808,
        'validation': 6894
    },
    't0_task_adaptation:wiqa_what_is_the_missing_first_step': {
        'test': 3003,
        'train': 29808,
        'validation': 6894
    },
    't0_task_adaptation:wiqa_what_might_be_the_first_step_of_the_process': {
        'test': 3003,
        'train': 29808,
        'validation': 6894
    },
    't0_task_adaptation:wiqa_what_might_be_the_last_step_of_the_process': {
        'test': 3003,
        'train': 29808,
        'validation': 6894
    },
    't0_task_adaptation:wiqa_which_of_the_following_is_the_supposed_perturbation':
        {
            'test': 3003,
            'train': 29808,
            'validation': 6894
        },
    't0_task_adaptation:xsum_DOC_boils_down_to_simple_idea_that': {
        'test': 11334,
        'train': 204045,
        'validation': 11332
    },
    't0_task_adaptation:xsum_DOC_given_above_write_one_sentence': {
        'test': 11334,
        'train': 204045,
        'validation': 11332
    },
    't0_task_adaptation:xsum_DOC_how_would_you_rephrase_few_words': {
        'test': 11334,
        'train': 204045,
        'validation': 11332
    },
    't0_task_adaptation:xsum_DOC_tldr': {
        'test': 11334,
        'train': 204045,
        'validation': 11332
    },
    't0_task_adaptation:xsum_DOC_write_summary_of_above': {
        'test': 11334,
        'train': 204045,
        'validation': 11332
    },
    't0_task_adaptation:xsum_article_DOC_summary': {
        'test': 11334,
        'train': 204045,
        'validation': 11332
    },
    't0_task_adaptation:xsum_college_roommate_asked_DOC_so_I_recap': {
        'test': 11334,
        'train': 204045,
        'validation': 11332
    },
    't0_task_adaptation:xsum_read_below_DOC_write_abstract': {
        'test': 11334,
        'train': 204045,
        'validation': 11332
    },
    't0_task_adaptation:xsum_summarize_DOC': {
        'test': 11334,
        'train': 204045,
        'validation': 11332
    },
    't0_task_adaptation:xsum_summarize_this_DOC_summary': {
        'test': 11334,
        'train': 204045,
        'validation': 11332
    },
    't0_task_adaptation:yelp_review_full_based_on_that': {
        'test': 50000,
        'train': 650000
    },
    't0_task_adaptation:yelp_review_full_format_rating': {
        'test': 50000,
        'train': 650000
    },
    't0_task_adaptation:yelp_review_full_format_score': {
        'test': 50000,
        'train': 650000
    },
    't0_task_adaptation:yelp_review_full_format_star': {
        'test': 50000,
        'train': 650000
    },
    't0_task_adaptation:yelp_review_full_on_a_scale': {
        'test': 50000,
        'train': 650000
    },
    't0_task_adaptation:yelp_review_full_so_i_would': {
        'test': 50000,
        'train': 650000
    },
    't0_task_adaptation:yelp_review_full_this_place': {
        'test': 50000,
        'train': 650000
    }
}

T0_TRAIN_TASK_METADATA = {
    't0_task_adaptation:adversarial_qa_dbert_answer_the_following_q': {
        'in_flan': False,
        'seq_len': {
            'max': 56,
            'mean': 4
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:adversarial_qa_dbert_based_on': {
        'in_flan': False,
        'seq_len': {
            'max': 56,
            'mean': 4
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:adversarial_qa_dbert_generate_question': {
        'in_flan': False,
        'seq_len': {
            'max': 22,
            'mean': 9
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:adversarial_qa_dbert_question_context_answer': {
        'in_flan': False,
        'seq_len': {
            'max': 56,
            'mean': 4
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:adversarial_qa_dbert_tell_what_it_is': {
        'in_flan': False,
        'seq_len': {
            'max': 56,
            'mean': 4
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:adversarial_qa_dbidaf_answer_the_following_q': {
        'in_flan': False,
        'seq_len': {
            'max': 33,
            'mean': 3
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:adversarial_qa_dbidaf_based_on': {
        'in_flan': False,
        'seq_len': {
            'max': 33,
            'mean': 3
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:adversarial_qa_dbidaf_generate_question': {
        'in_flan': False,
        'seq_len': {
            'max': 23,
            'mean': 9
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:adversarial_qa_dbidaf_question_context_answer': {
        'in_flan': False,
        'seq_len': {
            'max': 33,
            'mean': 3
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:adversarial_qa_dbidaf_tell_what_it_is': {
        'in_flan': False,
        'seq_len': {
            'max': 33,
            'mean': 3
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:adversarial_qa_droberta_answer_the_following_q': {
        'in_flan': False,
        'seq_len': {
            'max': 40,
            'mean': 4
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:adversarial_qa_droberta_based_on': {
        'in_flan': False,
        'seq_len': {
            'max': 40,
            'mean': 4
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:adversarial_qa_droberta_generate_question': {
        'in_flan': False,
        'seq_len': {
            'max': 21,
            'mean': 9
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:adversarial_qa_droberta_question_context_answer': {
        'in_flan': False,
        'seq_len': {
            'max': 40,
            'mean': 4
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:adversarial_qa_droberta_tell_what_it_is': {
        'in_flan': False,
        'seq_len': {
            'max': 40,
            'mean': 4
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:ag_news_classify': {
        'in_flan': True,
        'seq_len': {
            'max': 3,
            'mean': 2
        },
        'task_type': 't0_multiple_choice_separated_options'
    },
    't0_task_adaptation:ag_news_classify_question_first': {
        'in_flan': True,
        'seq_len': {
            'max': 3,
            'mean': 2
        },
        'task_type': 't0_multiple_choice_separated_options'
    },
    't0_task_adaptation:ag_news_classify_with_choices': {
        'in_flan': True,
        'seq_len': {
            'max': 3,
            'mean': 2
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:ag_news_classify_with_choices_question_first': {
        'in_flan': True,
        'seq_len': {
            'max': 3,
            'mean': 2
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:ag_news_recommend': {
        'in_flan': True,
        'seq_len': {
            'max': 2,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:ag_news_which_section': {
        'in_flan': True,
        'seq_len': {
            'max': 3,
            'mean': 2
        },
        'task_type': 't0_multiple_choice_separated_options'
    },
    't0_task_adaptation:ag_news_which_section_choices': {
        'in_flan': True,
        'seq_len': {
            'max': 3,
            'mean': 2
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:ai2_arc_ARC_Challenge_heres_a_problem': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:ai2_arc_ARC_Challenge_i_am_hesitating': {
        'in_flan': True,
        'seq_len': {
            'max': 25,
            'mean': 4
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:ai2_arc_ARC_Challenge_multiple_choice': {
        'in_flan': True,
        'seq_len': {
            'max': 25,
            'mean': 4
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:ai2_arc_ARC_Challenge_pick_false_options': {
        'in_flan': True,
        'seq_len': {
            'max': 49,
            'mean': 15
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:ai2_arc_ARC_Challenge_pick_the_most_correct_option': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:ai2_arc_ARC_Challenge_qa_options': {
        'in_flan': True,
        'seq_len': {
            'max': 25,
            'mean': 4
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:ai2_arc_ARC_Easy_heres_a_problem': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:ai2_arc_ARC_Easy_i_am_hesitating': {
        'in_flan': True,
        'seq_len': {
            'max': 23,
            'mean': 3
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:ai2_arc_ARC_Easy_multiple_choice': {
        'in_flan': True,
        'seq_len': {
            'max': 23,
            'mean': 3
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:ai2_arc_ARC_Easy_pick_false_options': {
        'in_flan': True,
        'seq_len': {
            'max': 45,
            'mean': 12
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:ai2_arc_ARC_Easy_pick_the_most_correct_option': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:ai2_arc_ARC_Easy_qa_options': {
        'in_flan': True,
        'seq_len': {
            'max': 23,
            'mean': 3
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:amazon_polarity_Is_this_product_review_positive': {
        'in_flan': False,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice_separated_options'
    },
    't0_task_adaptation:amazon_polarity_Is_this_review': {
        'in_flan': False,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:amazon_polarity_Is_this_review_negative': {
        'in_flan': False,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice_separated_options'
    },
    't0_task_adaptation:amazon_polarity_User_recommend_this_product': {
        'in_flan': False,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice_separated_options'
    },
    't0_task_adaptation:amazon_polarity_convey_negative_or_positive_sentiment':
        {
            'in_flan': False,
            'seq_len': {
                'max': 1,
                'mean': 1
            },
            'task_type': 't0_multiple_choice'
        },
    't0_task_adaptation:amazon_polarity_flattering_or_not': {
        'in_flan': False,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:amazon_polarity_negative_or_positive_tone': {
        'in_flan': False,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:amazon_polarity_user_satisfied': {
        'in_flan': False,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:amazon_polarity_would_you_buy': {
        'in_flan': False,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:anli_GPT_3_style_r1': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:anli_GPT_3_style_r1_score_eval': {
        'in_flan': True,
        'task_type': 't0_multiple_choice_score_eval'
    },
    't0_task_adaptation:anli_GPT_3_style_r2': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:anli_GPT_3_style_r2_score_eval': {
        'in_flan': True,
        'task_type': 't0_multiple_choice_score_eval'
    },
    't0_task_adaptation:anli_GPT_3_style_r3': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:anli_GPT_3_style_r3_score_eval': {
        'in_flan': True,
        'task_type': 't0_multiple_choice_score_eval'
    },
    't0_task_adaptation:anli_MNLI_crowdsource_r1': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:anli_MNLI_crowdsource_r1_score_eval': {
        'in_flan': True,
        'task_type': 't0_multiple_choice_score_eval'
    },
    't0_task_adaptation:anli_MNLI_crowdsource_r2': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:anli_MNLI_crowdsource_r2_score_eval': {
        'in_flan': True,
        'task_type': 't0_multiple_choice_score_eval'
    },
    't0_task_adaptation:anli_MNLI_crowdsource_r3': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:anli_MNLI_crowdsource_r3_score_eval': {
        'in_flan': True,
        'task_type': 't0_multiple_choice_score_eval'
    },
    't0_task_adaptation:anli_always_sometimes_never_r1': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:anli_always_sometimes_never_r1_score_eval': {
        'in_flan': True,
        'task_type': 't0_multiple_choice_score_eval'
    },
    't0_task_adaptation:anli_always_sometimes_never_r2': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:anli_always_sometimes_never_r2_score_eval': {
        'in_flan': True,
        'task_type': 't0_multiple_choice_score_eval'
    },
    't0_task_adaptation:anli_always_sometimes_never_r3': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:anli_always_sometimes_never_r3_score_eval': {
        'in_flan': True,
        'task_type': 't0_multiple_choice_score_eval'
    },
    't0_task_adaptation:anli_based_on_the_previous_passage_r1': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:anli_based_on_the_previous_passage_r1_score_eval': {
        'in_flan': True,
        'task_type': 't0_multiple_choice_score_eval'
    },
    't0_task_adaptation:anli_based_on_the_previous_passage_r2': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:anli_based_on_the_previous_passage_r2_score_eval': {
        'in_flan': True,
        'task_type': 't0_multiple_choice_score_eval'
    },
    't0_task_adaptation:anli_based_on_the_previous_passage_r3': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:anli_based_on_the_previous_passage_r3_score_eval': {
        'in_flan': True,
        'task_type': 't0_multiple_choice_score_eval'
    },
    't0_task_adaptation:anli_can_we_infer_r1': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:anli_can_we_infer_r1_score_eval': {
        'in_flan': True,
        'task_type': 't0_multiple_choice_score_eval'
    },
    't0_task_adaptation:anli_can_we_infer_r2': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:anli_can_we_infer_r2_score_eval': {
        'in_flan': True,
        'task_type': 't0_multiple_choice_score_eval'
    },
    't0_task_adaptation:anli_can_we_infer_r3': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:anli_can_we_infer_r3_score_eval': {
        'in_flan': True,
        'task_type': 't0_multiple_choice_score_eval'
    },
    't0_task_adaptation:anli_claim_true_false_inconclusive_r1': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:anli_claim_true_false_inconclusive_r1_score_eval': {
        'in_flan': True,
        'task_type': 't0_multiple_choice_score_eval'
    },
    't0_task_adaptation:anli_claim_true_false_inconclusive_r2': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:anli_claim_true_false_inconclusive_r2_score_eval': {
        'in_flan': True,
        'task_type': 't0_multiple_choice_score_eval'
    },
    't0_task_adaptation:anli_claim_true_false_inconclusive_r3': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:anli_claim_true_false_inconclusive_r3_score_eval': {
        'in_flan': True,
        'task_type': 't0_multiple_choice_score_eval'
    },
    't0_task_adaptation:anli_consider_always_sometimes_never_r1': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:anli_consider_always_sometimes_never_r1_score_eval': {
        'in_flan': True,
        'task_type': 't0_multiple_choice_score_eval'
    },
    't0_task_adaptation:anli_consider_always_sometimes_never_r2': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:anli_consider_always_sometimes_never_r2_score_eval': {
        'in_flan': True,
        'task_type': 't0_multiple_choice_score_eval'
    },
    't0_task_adaptation:anli_consider_always_sometimes_never_r3': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:anli_consider_always_sometimes_never_r3_score_eval': {
        'in_flan': True,
        'task_type': 't0_multiple_choice_score_eval'
    },
    't0_task_adaptation:anli_does_it_follow_that_r1': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:anli_does_it_follow_that_r1_score_eval': {
        'in_flan': True,
        'task_type': 't0_multiple_choice_score_eval'
    },
    't0_task_adaptation:anli_does_it_follow_that_r2': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:anli_does_it_follow_that_r2_score_eval': {
        'in_flan': True,
        'task_type': 't0_multiple_choice_score_eval'
    },
    't0_task_adaptation:anli_does_it_follow_that_r3': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:anli_does_it_follow_that_r3_score_eval': {
        'in_flan': True,
        'task_type': 't0_multiple_choice_score_eval'
    },
    't0_task_adaptation:anli_does_this_imply_r1': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:anli_does_this_imply_r1_score_eval': {
        'in_flan': True,
        'task_type': 't0_multiple_choice_score_eval'
    },
    't0_task_adaptation:anli_does_this_imply_r2': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:anli_does_this_imply_r2_score_eval': {
        'in_flan': True,
        'task_type': 't0_multiple_choice_score_eval'
    },
    't0_task_adaptation:anli_does_this_imply_r3': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:anli_does_this_imply_r3_score_eval': {
        'in_flan': True,
        'task_type': 't0_multiple_choice_score_eval'
    },
    't0_task_adaptation:anli_guaranteed_possible_impossible_r1': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:anli_guaranteed_possible_impossible_r1_score_eval': {
        'in_flan': True,
        'task_type': 't0_multiple_choice_score_eval'
    },
    't0_task_adaptation:anli_guaranteed_possible_impossible_r2': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:anli_guaranteed_possible_impossible_r2_score_eval': {
        'in_flan': True,
        'task_type': 't0_multiple_choice_score_eval'
    },
    't0_task_adaptation:anli_guaranteed_possible_impossible_r3': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:anli_guaranteed_possible_impossible_r3_score_eval': {
        'in_flan': True,
        'task_type': 't0_multiple_choice_score_eval'
    },
    't0_task_adaptation:anli_guaranteed_true_r1': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:anli_guaranteed_true_r1_score_eval': {
        'in_flan': True,
        'task_type': 't0_multiple_choice_score_eval'
    },
    't0_task_adaptation:anli_guaranteed_true_r2': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:anli_guaranteed_true_r2_score_eval': {
        'in_flan': True,
        'task_type': 't0_multiple_choice_score_eval'
    },
    't0_task_adaptation:anli_guaranteed_true_r3': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:anli_guaranteed_true_r3_score_eval': {
        'in_flan': True,
        'task_type': 't0_multiple_choice_score_eval'
    },
    't0_task_adaptation:anli_justified_in_saying_r1': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:anli_justified_in_saying_r1_score_eval': {
        'in_flan': True,
        'task_type': 't0_multiple_choice_score_eval'
    },
    't0_task_adaptation:anli_justified_in_saying_r2': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:anli_justified_in_saying_r2_score_eval': {
        'in_flan': True,
        'task_type': 't0_multiple_choice_score_eval'
    },
    't0_task_adaptation:anli_justified_in_saying_r3': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:anli_justified_in_saying_r3_score_eval': {
        'in_flan': True,
        'task_type': 't0_multiple_choice_score_eval'
    },
    't0_task_adaptation:anli_must_be_true_r1': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:anli_must_be_true_r1_score_eval': {
        'in_flan': True,
        'task_type': 't0_multiple_choice_score_eval'
    },
    't0_task_adaptation:anli_must_be_true_r2': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:anli_must_be_true_r2_score_eval': {
        'in_flan': True,
        'task_type': 't0_multiple_choice_score_eval'
    },
    't0_task_adaptation:anli_must_be_true_r3': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:anli_must_be_true_r3_score_eval': {
        'in_flan': True,
        'task_type': 't0_multiple_choice_score_eval'
    },
    't0_task_adaptation:anli_should_assume_r1': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:anli_should_assume_r1_score_eval': {
        'in_flan': True,
        'task_type': 't0_multiple_choice_score_eval'
    },
    't0_task_adaptation:anli_should_assume_r2': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:anli_should_assume_r2_score_eval': {
        'in_flan': True,
        'task_type': 't0_multiple_choice_score_eval'
    },
    't0_task_adaptation:anli_should_assume_r3': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:anli_should_assume_r3_score_eval': {
        'in_flan': True,
        'task_type': 't0_multiple_choice_score_eval'
    },
    't0_task_adaptation:anli_take_the_following_as_truth_r1': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:anli_take_the_following_as_truth_r1_score_eval': {
        'in_flan': True,
        'task_type': 't0_multiple_choice_score_eval'
    },
    't0_task_adaptation:anli_take_the_following_as_truth_r2': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:anli_take_the_following_as_truth_r2_score_eval': {
        'in_flan': True,
        'task_type': 't0_multiple_choice_score_eval'
    },
    't0_task_adaptation:anli_take_the_following_as_truth_r3': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:anli_take_the_following_as_truth_r3_score_eval': {
        'in_flan': True,
        'task_type': 't0_multiple_choice_score_eval'
    },
    't0_task_adaptation:app_reviews_categorize_rating_using_review': {
        'in_flan': False,
        'seq_len': {
            'max': 3,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:app_reviews_convert_to_rating': {
        'in_flan': False,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:app_reviews_convert_to_star_rating': {
        'in_flan': False,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:app_reviews_generate_review': {
        'in_flan': False,
        'seq_len': {
            'max': 138,
            'mean': 22
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:cnn_dailymail_3.0.0_2_or_3_sentences': {
        'in_flan': True,
        'seq_len': {
            'max': 66,
            'mean': 42
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:cnn_dailymail_3.0.0_generate_story': {
        'in_flan': True,
        'seq_len': {
            'max': 391,
            'mean': 320
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:cnn_dailymail_3.0.0_news_card_view': {
        'in_flan': True,
        'seq_len': {
            'max': 66,
            'mean': 42
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:cnn_dailymail_3.0.0_news_stock': {
        'in_flan': True,
        'seq_len': {
            'max': 66,
            'mean': 42
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:cnn_dailymail_3.0.0_news_summary': {
        'in_flan': True,
        'seq_len': {
            'max': 66,
            'mean': 42
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:cnn_dailymail_3.0.0_spice_up_story': {
        'in_flan': True,
        'seq_len': {
            'max': 391,
            'mean': 320
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:cnn_dailymail_3.0.0_sum_in_brief': {
        'in_flan': True,
        'seq_len': {
            'max': 66,
            'mean': 42
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:cnn_dailymail_3.0.0_tldr_summary': {
        'in_flan': True,
        'seq_len': {
            'max': 66,
            'mean': 42
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:cnn_dailymail_3.0.0_write_an_outline': {
        'in_flan': True,
        'seq_len': {
            'max': 66,
            'mean': 42
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:common_gen_Example_prompt': {
        'in_flan': True,
        'seq_len': {
            'max': 12,
            'mean': 7
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:common_gen_Given_concepts_type_1': {
        'in_flan': True,
        'seq_len': {
            'max': 12,
            'mean': 7
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:common_gen_Given_concepts_type_2': {
        'in_flan': True,
        'seq_len': {
            'max': 12,
            'mean': 7
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:common_gen_Put_together': {
        'in_flan': True,
        'seq_len': {
            'max': 12,
            'mean': 7
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:common_gen_choice_in_concept_centric_sentence_generation':
        {
            'in_flan': True,
            'seq_len': {
                'max': 12,
                'mean': 7
            },
            'task_type': 't0_question_answer'
        },
    't0_task_adaptation:common_gen_random_task_template_prompt': {
        'in_flan': True,
        'seq_len': {
            'max': 12,
            'mean': 7
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:common_gen_sentence_to_concepts': {
        'in_flan': True,
        'seq_len': {
            'max': 3,
            'mean': 3
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:common_gen_topic_to_sentence': {
        'in_flan': True,
        'seq_len': {
            'max': 12,
            'mean': 7
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:common_gen_topics_from_the_sentence': {
        'in_flan': True,
        'seq_len': {
            'max': 3,
            'mean': 3
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:cos_e_v1.11_aligned_with_common_sense': {
        'in_flan': False,
        'seq_len': {
            'max': 29,
            'mean': 7
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:cos_e_v1.11_description_question_option_id': {
        'in_flan': False,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:cos_e_v1.11_description_question_option_text': {
        'in_flan': False,
        'seq_len': {
            'max': 4,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:cos_e_v1.11_explain_why_human': {
        'in_flan': False,
        'seq_len': {
            'max': 29,
            'mean': 7
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:cos_e_v1.11_generate_explanation_given_text': {
        'in_flan': False,
        'seq_len': {
            'max': 29,
            'mean': 7
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:cos_e_v1.11_i_think': {
        'in_flan': False,
        'seq_len': {
            'max': 29,
            'mean': 7
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:cos_e_v1.11_question_description_option_id': {
        'in_flan': False,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:cos_e_v1.11_question_description_option_text': {
        'in_flan': False,
        'seq_len': {
            'max': 4,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:cos_e_v1.11_question_option_description_id': {
        'in_flan': False,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:cos_e_v1.11_question_option_description_text': {
        'in_flan': False,
        'seq_len': {
            'max': 4,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:cos_e_v1.11_rationale': {
        'in_flan': False,
        'seq_len': {
            'max': 29,
            'mean': 7
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:cosmos_qa_context_answer_to_question': {
        'in_flan': True,
        'seq_len': {
            'max': 22,
            'mean': 11
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:cosmos_qa_context_description_question_answer_id': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:cosmos_qa_context_description_question_answer_text': {
        'in_flan': True,
        'seq_len': {
            'max': 32,
            'mean': 9
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:cosmos_qa_context_description_question_text': {
        'in_flan': True,
        'seq_len': {
            'max': 32,
            'mean': 9
        },
        'task_type': 't0_multiple_choice_separated_options'
    },
    't0_task_adaptation:cosmos_qa_context_question_description_answer_id': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:cosmos_qa_context_question_description_answer_text': {
        'in_flan': True,
        'seq_len': {
            'max': 32,
            'mean': 9
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:cosmos_qa_context_question_description_text': {
        'in_flan': True,
        'seq_len': {
            'max': 32,
            'mean': 9
        },
        'task_type': 't0_multiple_choice_separated_options'
    },
    't0_task_adaptation:cosmos_qa_description_context_question_answer_id': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:cosmos_qa_description_context_question_answer_text': {
        'in_flan': True,
        'seq_len': {
            'max': 32,
            'mean': 9
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:cosmos_qa_description_context_question_text': {
        'in_flan': True,
        'seq_len': {
            'max': 32,
            'mean': 9
        },
        'task_type': 't0_multiple_choice_separated_options'
    },
    't0_task_adaptation:cosmos_qa_no_prompt_id': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:cosmos_qa_no_prompt_text': {
        'in_flan': True,
        'seq_len': {
            'max': 32,
            'mean': 9
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:cosmos_qa_only_question_answer': {
        'in_flan': True,
        'seq_len': {
            'max': 32,
            'mean': 9
        },
        'task_type': 't0_multiple_choice_separated_options'
    },
    't0_task_adaptation:dbpedia_14_given_a_choice_of_categories_': {
        'in_flan': False,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:dbpedia_14_given_a_list_of_category_what_does_the_title_belong_to':
        {
            'in_flan': False,
            'seq_len': {
                'max': 1,
                'mean': 1
            },
            'task_type': 't0_multiple_choice'
        },
    't0_task_adaptation:dbpedia_14_given_list_what_category_does_the_paragraph_belong_to':
        {
            'in_flan': False,
            'seq_len': {
                'max': 1,
                'mean': 1
            },
            'task_type': 't0_multiple_choice'
        },
    't0_task_adaptation:dbpedia_14_pick_one_category_for_the_following_text': {
        'in_flan': False,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:dream_answer_to_dialogue': {
        'in_flan': False,
        'seq_len': {
            'max': 349,
            'mean': 124
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:dream_baseline': {
        'in_flan': False,
        'seq_len': {
            'max': 12,
            'mean': 4
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:dream_generate_first_utterance': {
        'in_flan': False,
        'seq_len': {
            'max': 133,
            'mean': 14
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:dream_generate_last_utterance': {
        'in_flan': False,
        'seq_len': {
            'max': 44,
            'mean': 12
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:dream_read_the_following_conversation_and_answer_the_question':
        {
            'in_flan': False,
            'seq_len': {
                'max': 12,
                'mean': 4
            },
            'task_type': 't0_multiple_choice'
        },
    't0_task_adaptation:duorc_ParaphraseRC_answer_question': {
        'in_flan': False,
        'seq_len': {
            'max': 15,
            'mean': 2
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:duorc_ParaphraseRC_build_story_around_qa': {
        'in_flan': False,
        'seq_len': {
            'max': 364,
            'mean': 334
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:duorc_ParaphraseRC_decide_worth_it': {
        'in_flan': False,
        'seq_len': {
            'max': 15,
            'mean': 3
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:duorc_ParaphraseRC_extract_answer': {
        'in_flan': False,
        'seq_len': {
            'max': 15,
            'mean': 2
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:duorc_ParaphraseRC_generate_question': {
        'in_flan': False,
        'seq_len': {
            'max': 24,
            'mean': 7
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:duorc_ParaphraseRC_generate_question_by_answer': {
        'in_flan': False,
        'seq_len': {
            'max': 24,
            'mean': 7
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:duorc_ParaphraseRC_movie_director': {
        'in_flan': False,
        'seq_len': {
            'max': 15,
            'mean': 2
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:duorc_ParaphraseRC_question_answering': {
        'in_flan': False,
        'seq_len': {
            'max': 15,
            'mean': 3
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:duorc_ParaphraseRC_title_generation': {
        'in_flan': False,
        'seq_len': {
            'max': 7,
            'mean': 2
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:duorc_SelfRC_answer_question': {
        'in_flan': False,
        'seq_len': {
            'max': 15,
            'mean': 2
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:duorc_SelfRC_build_story_around_qa': {
        'in_flan': False,
        'seq_len': {
            'max': 357,
            'mean': 330
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:duorc_SelfRC_decide_worth_it': {
        'in_flan': False,
        'seq_len': {
            'max': 15,
            'mean': 2
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:duorc_SelfRC_extract_answer': {
        'in_flan': False,
        'seq_len': {
            'max': 15,
            'mean': 2
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:duorc_SelfRC_generate_question': {
        'in_flan': False,
        'seq_len': {
            'max': 24,
            'mean': 7
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:duorc_SelfRC_generate_question_by_answer': {
        'in_flan': False,
        'seq_len': {
            'max': 24,
            'mean': 7
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:duorc_SelfRC_movie_director': {
        'in_flan': False,
        'seq_len': {
            'max': 15,
            'mean': 2
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:duorc_SelfRC_question_answering': {
        'in_flan': False,
        'seq_len': {
            'max': 15,
            'mean': 2
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:duorc_SelfRC_title_generation': {
        'in_flan': False,
        'seq_len': {
            'max': 7,
            'mean': 2
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:gigaword_TLDR': {
        'in_flan': True,
        'seq_len': {
            'max': 13,
            'mean': 7
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:gigaword_first_sentence_title': {
        'in_flan': True,
        'seq_len': {
            'max': 13,
            'mean': 7
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:gigaword_generate_summary_for_this': {
        'in_flan': True,
        'seq_len': {
            'max': 13,
            'mean': 7
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:gigaword_in_a_nutshell': {
        'in_flan': True,
        'seq_len': {
            'max': 13,
            'mean': 7
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:gigaword_make_a_title': {
        'in_flan': True,
        'seq_len': {
            'max': 13,
            'mean': 7
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:gigaword_reverse_writing': {
        'in_flan': True,
        'seq_len': {
            'max': 50,
            'mean': 31
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:gigaword_write_a_title_for_this_sentence': {
        'in_flan': True,
        'seq_len': {
            'max': 13,
            'mean': 7
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:gigaword_write_an_article': {
        'in_flan': True,
        'seq_len': {
            'max': 50,
            'mean': 31
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:gigaword_write_its_sentence': {
        'in_flan': True,
        'seq_len': {
            'max': 13,
            'mean': 7
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:glue_mrpc_equivalent': {
        'in_flan': True,
        'seq_len': {
            'max': 2,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:glue_mrpc_generate_paraphrase': {
        'in_flan': True,
        'seq_len': {
            'max': 35,
            'mean': 22
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:glue_mrpc_generate_sentence': {
        'in_flan': True,
        'seq_len': {
            'max': 35,
            'mean': 22
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:glue_mrpc_paraphrase': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice_separated_options'
    },
    't0_task_adaptation:glue_mrpc_replace': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice_separated_options'
    },
    't0_task_adaptation:glue_mrpc_same_thing': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice_separated_options'
    },
    't0_task_adaptation:glue_mrpc_want_to_know': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice_separated_options'
    },
    't0_task_adaptation:glue_qqp_answer': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice_separated_options'
    },
    't0_task_adaptation:glue_qqp_duplicate': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice_separated_options'
    },
    't0_task_adaptation:glue_qqp_duplicate_or_not': {
        'in_flan': True,
        'seq_len': {
            'max': 2,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:glue_qqp_meaning': {
        'in_flan': True,
        'seq_len': {
            'max': 0,
            'mean': 0
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:glue_qqp_quora': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice_separated_options'
    },
    't0_task_adaptation:glue_qqp_same_thing': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice_separated_options'
    },
    't0_task_adaptation:hellaswag_Appropriate_continuation_Yes_or_No': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:hellaswag_Open_ended_completion': {
        'in_flan': True,
        'seq_len': {
            'max': 39,
            'mean': 12
        },
        'task_type': 't0_multiple_choice_separated_options'
    },
    't0_task_adaptation:hellaswag_Open_ended_start': {
        'in_flan': True,
        'seq_len': {
            'max': 52,
            'mean': 22
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:hellaswag_Predict_ending_with_hint': {
        'in_flan': True,
        'seq_len': {
            'max': 39,
            'mean': 12
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:hellaswag_Predict_ending_with_hint_score_eval': {
        'in_flan': True,
        'task_type': 't0_multiple_choice_score_eval'
    },
    't0_task_adaptation:hellaswag_Randomized_prompts_template': {
        'in_flan': True,
        'seq_len': {
            'max': 39,
            'mean': 12
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:hellaswag_Randomized_prompts_template_score_eval': {
        'in_flan': True,
        'task_type': 't0_multiple_choice_score_eval'
    },
    't0_task_adaptation:hellaswag_Reversed_appropriate_continuation_Yes_or_No':
        {
            'in_flan': True,
            'seq_len': {
                'max': 1,
                'mean': 1
            },
            'task_type': 't0_multiple_choice'
        },
    't0_task_adaptation:hellaswag_Topic_of_the_context': {
        'in_flan': True,
        'seq_len': {
            'max': 5,
            'mean': 2
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:hellaswag_Topic_without_the_ending_answer': {
        'in_flan': True,
        'seq_len': {
            'max': 5,
            'mean': 2
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:hellaswag_complete_first_then': {
        'in_flan': True,
        'seq_len': {
            'max': 39,
            'mean': 12
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:hellaswag_complete_first_then_score_eval': {
        'in_flan': True,
        'task_type': 't0_multiple_choice_score_eval'
    },
    't0_task_adaptation:hellaswag_how_ends': {
        'in_flan': True,
        'seq_len': {
            'max': 2,
            'mean': 2
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:hellaswag_if_begins_how_continues': {
        'in_flan': True,
        'seq_len': {
            'max': 2,
            'mean': 2
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:hellaswag_if_begins_how_continues_score_eval': {
        'in_flan': True,
        'task_type': 't0_multiple_choice_score_eval'
    },
    't0_task_adaptation:imdb_Movie_Expressed_Sentiment': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice_separated_options'
    },
    't0_task_adaptation:imdb_Movie_Expressed_Sentiment_2': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice_separated_options'
    },
    't0_task_adaptation:imdb_Negation_template_for_positive_and_negative': {
        'in_flan': True,
        'seq_len': {
            'max': 2,
            'mean': 2
        },
        'task_type': 't0_multiple_choice_separated_options'
    },
    't0_task_adaptation:imdb_Reviewer_Enjoyment': {
        'in_flan': True,
        'seq_len': {
            'max': 3,
            'mean': 3
        },
        'task_type': 't0_multiple_choice_separated_options'
    },
    't0_task_adaptation:imdb_Reviewer_Enjoyment_Yes_No': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice_separated_options'
    },
    't0_task_adaptation:imdb_Reviewer_Expressed_Sentiment': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice_separated_options'
    },
    't0_task_adaptation:imdb_Reviewer_Opinion_bad_good_choices': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:imdb_Reviewer_Sentiment_Feeling': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice_separated_options'
    },
    't0_task_adaptation:imdb_Sentiment_with_choices_': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:imdb_Text_Expressed_Sentiment': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice_separated_options'
    },
    't0_task_adaptation:imdb_Writer_Expressed_Sentiment': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice_separated_options'
    },
    't0_task_adaptation:kilt_tasks_hotpotqa_combining_facts': {
        'in_flan': False,
        'seq_len': {
            'max': 12,
            'mean': 2
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:kilt_tasks_hotpotqa_complex_question': {
        'in_flan': False,
        'seq_len': {
            'max': 12,
            'mean': 2
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:kilt_tasks_hotpotqa_final_exam': {
        'in_flan': False,
        'seq_len': {
            'max': 12,
            'mean': 2
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:kilt_tasks_hotpotqa_formulate': {
        'in_flan': False,
        'seq_len': {
            'max': 12,
            'mean': 2
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:kilt_tasks_hotpotqa_straighforward_qa': {
        'in_flan': False,
        'seq_len': {
            'max': 12,
            'mean': 2
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:multi_news_distill': {
        'in_flan': True,
        'seq_len': {
            'max': 339,
            'mean': 210
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:multi_news_expand_reverse_task_': {
        'in_flan': True,
        'seq_len': {
            'max': 367,
            'mean': 254
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:multi_news_summarize': {
        'in_flan': True,
        'seq_len': {
            'max': 339,
            'mean': 210
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:multi_news_summary_scenario': {
        'in_flan': True,
        'seq_len': {
            'max': 339,
            'mean': 210
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:multi_news_synthesize': {
        'in_flan': True,
        'seq_len': {
            'max': 339,
            'mean': 210
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:multi_news_what_are_the_key_points': {
        'in_flan': True,
        'seq_len': {
            'max': 339,
            'mean': 210
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:openbookqa_main_choices': {
        'in_flan': True,
        'seq_len': {
            'max': 13,
            'mean': 3
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:openbookqa_main_choose_an_answer_with_options': {
        'in_flan': True,
        'seq_len': {
            'max': 13,
            'mean': 3
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:openbookqa_main_only_options': {
        'in_flan': True,
        'seq_len': {
            'max': 13,
            'mean': 3
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:openbookqa_main_pick_answer_with_options': {
        'in_flan': True,
        'seq_len': {
            'max': 13,
            'mean': 3
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:openbookqa_main_pick_using_id': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:openbookqa_main_which_correct': {
        'in_flan': True,
        'seq_len': {
            'max': 13,
            'mean': 3
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:openbookqa_main_which_correct_inverse': {
        'in_flan': True,
        'seq_len': {
            'max': 13,
            'mean': 3
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:paws_labeled_final_Concatenation': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:paws_labeled_final_Concatenation_no_label': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice_separated_options'
    },
    't0_task_adaptation:paws_labeled_final_Meaning': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:paws_labeled_final_Meaning_no_label': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice_separated_options'
    },
    't0_task_adaptation:paws_labeled_final_PAWS_ANLI_GPT3': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:paws_labeled_final_PAWS_ANLI_GPT3_no_label': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice_separated_options'
    },
    't0_task_adaptation:paws_labeled_final_Rewrite': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:paws_labeled_final_Rewrite_no_label': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice_separated_options'
    },
    't0_task_adaptation:paws_labeled_final_context_question': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:paws_labeled_final_context_question_no_label': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice_separated_options'
    },
    't0_task_adaptation:paws_labeled_final_paraphrase_task': {
        'in_flan': True,
        'seq_len': {
            'max': 33,
            'mean': 21
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:paws_labeled_final_task_description_no_label': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice_separated_options'
    },
    't0_task_adaptation:piqa_Correct_the_solution': {
        'in_flan': True,
        'seq_len': {
            'max': 99,
            'mean': 18
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:piqa_Correct_the_solution_if_false_from_sol_1': {
        'in_flan': True,
        'seq_len': {
            'max': 106,
            'mean': 25
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:piqa_Correct_the_solution_if_false_from_sol_2': {
        'in_flan': True,
        'seq_len': {
            'max': 106,
            'mean': 25
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:piqa_Does_this_solution_make_sense_sol1': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:piqa_Does_this_solution_make_sense_sol2': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice_separated_options'
    },
    't0_task_adaptation:piqa_choose_the_most_appropriate_solution': {
        'in_flan': True,
        'seq_len': {
            'max': 2,
            'mean': 2
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:piqa_finish_sentence_with_correct_choice': {
        'in_flan': True,
        'seq_len': {
            'max': 99,
            'mean': 18
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:piqa_no_prompt_needed': {
        'in_flan': True,
        'seq_len': {
            'max': 99,
            'mean': 18
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:piqa_pick_correct_choice_index': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:piqa_pick_correct_choice_with_choice_given_before_goal':
        {
            'in_flan': True,
            'seq_len': {
                'max': 99,
                'mean': 18
            },
            'task_type': 't0_multiple_choice'
        },
    't0_task_adaptation:piqa_what_is_the_correct_ending': {
        'in_flan': True,
        'seq_len': {
            'max': 99,
            'mean': 18
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:qasc_is_correct_1': {
        'in_flan': False,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice_separated_options'
    },
    't0_task_adaptation:qasc_is_correct_2': {
        'in_flan': False,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice_separated_options'
    },
    't0_task_adaptation:qasc_qa_with_combined_facts_1': {
        'in_flan': False,
        'seq_len': {
            'max': 8,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:qasc_qa_with_separated_facts_1': {
        'in_flan': False,
        'seq_len': {
            'max': 8,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:qasc_qa_with_separated_facts_2': {
        'in_flan': False,
        'seq_len': {
            'max': 8,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:qasc_qa_with_separated_facts_3': {
        'in_flan': False,
        'seq_len': {
            'max': 8,
            'mean': 1
        },
        'task_type': 't0_multiple_choice_separated_options'
    },
    't0_task_adaptation:qasc_qa_with_separated_facts_4': {
        'in_flan': False,
        'seq_len': {
            'max': 8,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:qasc_qa_with_separated_facts_5': {
        'in_flan': False,
        'seq_len': {
            'max': 8,
            'mean': 1
        },
        'task_type': 't0_multiple_choice_separated_options'
    },
    't0_task_adaptation:quail_context_description_question_answer_id': {
        'in_flan': False,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:quail_context_description_question_answer_text': {
        'in_flan': False,
        'seq_len': {
            'max': 12,
            'mean': 3
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:quail_context_description_question_text': {
        'in_flan': False,
        'seq_len': {
            'max': 12,
            'mean': 3
        },
        'task_type': 't0_multiple_choice_separated_options'
    },
    't0_task_adaptation:quail_context_question_answer_description_id': {
        'in_flan': False,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:quail_context_question_answer_description_text': {
        'in_flan': False,
        'seq_len': {
            'max': 12,
            'mean': 3
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:quail_context_question_description_answer_id': {
        'in_flan': False,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:quail_context_question_description_answer_text': {
        'in_flan': False,
        'seq_len': {
            'max': 12,
            'mean': 3
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:quail_context_question_description_text': {
        'in_flan': False,
        'seq_len': {
            'max': 12,
            'mean': 3
        },
        'task_type': 't0_multiple_choice_separated_options'
    },
    't0_task_adaptation:quail_description_context_question_answer_id': {
        'in_flan': False,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:quail_description_context_question_answer_text': {
        'in_flan': False,
        'seq_len': {
            'max': 12,
            'mean': 3
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:quail_description_context_question_text': {
        'in_flan': False,
        'seq_len': {
            'max': 12,
            'mean': 3
        },
        'task_type': 't0_multiple_choice_separated_options'
    },
    't0_task_adaptation:quail_no_prompt_id': {
        'in_flan': False,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:quail_no_prompt_text': {
        'in_flan': False,
        'seq_len': {
            'max': 12,
            'mean': 3
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:quarel_choose_between': {
        'in_flan': False,
        'seq_len': {
            'max': 5,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:quarel_do_not_use': {
        'in_flan': False,
        'seq_len': {
            'max': 5,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:quarel_heres_a_story': {
        'in_flan': False,
        'seq_len': {
            'max': 5,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:quarel_logic_test': {
        'in_flan': False,
        'seq_len': {
            'max': 5,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:quarel_testing_students': {
        'in_flan': False,
        'seq_len': {
            'max': 5,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:quartz_answer_question_based_on': {
        'in_flan': False,
        'seq_len': {
            'max': 12,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:quartz_answer_question_below': {
        'in_flan': False,
        'seq_len': {
            'max': 12,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:quartz_given_the_fact_answer_the_q': {
        'in_flan': False,
        'seq_len': {
            'max': 12,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:quartz_having_read_above_passage': {
        'in_flan': False,
        'seq_len': {
            'max': 12,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:quartz_paragraph_question_plain_concat': {
        'in_flan': False,
        'seq_len': {
            'max': 12,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:quartz_read_passage_below_choose': {
        'in_flan': False,
        'seq_len': {
            'max': 12,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:quartz_use_info_from_paragraph_question': {
        'in_flan': False,
        'seq_len': {
            'max': 12,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:quartz_use_info_from_question_paragraph': {
        'in_flan': False,
        'seq_len': {
            'max': 12,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:quoref_Answer_Friend_Question': {
        'in_flan': False,
        'seq_len': {
            'max': 9,
            'mean': 1
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:quoref_Answer_Question_Given_Context': {
        'in_flan': False,
        'seq_len': {
            'max': 9,
            'mean': 1
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:quoref_Answer_Test': {
        'in_flan': False,
        'seq_len': {
            'max': 9,
            'mean': 1
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:quoref_Context_Contains_Answer': {
        'in_flan': False,
        'seq_len': {
            'max': 9,
            'mean': 1
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:quoref_Find_Answer': {
        'in_flan': False,
        'seq_len': {
            'max': 9,
            'mean': 1
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:quoref_Found_Context_Online': {
        'in_flan': False,
        'seq_len': {
            'max': 9,
            'mean': 1
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:quoref_Given_Context_Answer_Question': {
        'in_flan': False,
        'seq_len': {
            'max': 9,
            'mean': 1
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:quoref_Guess_Answer': {
        'in_flan': False,
        'seq_len': {
            'max': 9,
            'mean': 1
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:quoref_Guess_Title_For_Context': {
        'in_flan': False,
        'seq_len': {
            'max': 6,
            'mean': 2
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:quoref_Read_And_Extract_': {
        'in_flan': False,
        'seq_len': {
            'max': 9,
            'mean': 1
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:quoref_What_Is_The_Answer': {
        'in_flan': False,
        'seq_len': {
            'max': 9,
            'mean': 1
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:race_high_Is_this_the_right_answer': {
        'in_flan': False,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:race_high_Read_the_article_and_answer_the_question_no_option_':
        {
            'in_flan': False,
            'seq_len': {
                'max': 21,
                'mean': 6
            },
            'task_type': 't0_multiple_choice_separated_options'
        },
    't0_task_adaptation:race_high_Select_the_best_answer': {
        'in_flan': False,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:race_high_Select_the_best_answer_generate_span_': {
        'in_flan': False,
        'seq_len': {
            'max': 21,
            'mean': 6
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:race_high_Select_the_best_answer_no_instructions_': {
        'in_flan': False,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:race_high_Taking_a_test': {
        'in_flan': False,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:race_high_Write_a_multi_choice_question_for_the_following_article':
        {
            'in_flan': False,
            'seq_len': {
                'max': 75,
                'mean': 34
            },
            'task_type': 't0_question_answer'
        },
    't0_task_adaptation:race_high_Write_a_multi_choice_question_options_given_':
        {
            'in_flan': False,
            'seq_len': {
                'max': 20,
                'mean': 10
            },
            'task_type': 't0_question_answer'
        },
    't0_task_adaptation:race_middle_Is_this_the_right_answer': {
        'in_flan': False,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:race_middle_Read_the_article_and_answer_the_question_no_option_':
        {
            'in_flan': False,
            'seq_len': {
                'max': 14,
                'mean': 4
            },
            'task_type': 't0_multiple_choice_separated_options'
        },
    't0_task_adaptation:race_middle_Select_the_best_answer': {
        'in_flan': False,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:race_middle_Select_the_best_answer_generate_span_': {
        'in_flan': False,
        'seq_len': {
            'max': 14,
            'mean': 4
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:race_middle_Select_the_best_answer_no_instructions_': {
        'in_flan': False,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:race_middle_Taking_a_test': {
        'in_flan': False,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:race_middle_Write_a_multi_choice_question_for_the_following_article':
        {
            'in_flan': False,
            'seq_len': {
                'max': 52,
                'mean': 25
            },
            'task_type': 't0_question_answer'
        },
    't0_task_adaptation:race_middle_Write_a_multi_choice_question_options_given_':
        {
            'in_flan': False,
            'seq_len': {
                'max': 29,
                'mean': 9
            },
            'task_type': 't0_question_answer'
        },
    't0_task_adaptation:ropes_background_new_situation_answer': {
        'in_flan': False,
        'seq_len': {
            'max': 2,
            'mean': 1
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:ropes_background_situation_middle': {
        'in_flan': False,
        'seq_len': {
            'max': 2,
            'mean': 1
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:ropes_given_background_situation': {
        'in_flan': False,
        'seq_len': {
            'max': 2,
            'mean': 1
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:ropes_new_situation_background_answer': {
        'in_flan': False,
        'seq_len': {
            'max': 2,
            'mean': 1
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:ropes_plain_background_situation': {
        'in_flan': False,
        'seq_len': {
            'max': 2,
            'mean': 1
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:ropes_plain_bottom_hint': {
        'in_flan': False,
        'seq_len': {
            'max': 2,
            'mean': 1
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:ropes_plain_no_background': {
        'in_flan': False,
        'seq_len': {
            'max': 2,
            'mean': 1
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:ropes_prompt_beginning': {
        'in_flan': False,
        'seq_len': {
            'max': 2,
            'mean': 1
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:ropes_prompt_bottom_hint_beginning': {
        'in_flan': False,
        'seq_len': {
            'max': 2,
            'mean': 1
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:ropes_prompt_bottom_no_hint': {
        'in_flan': False,
        'seq_len': {
            'max': 2,
            'mean': 1
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:ropes_prompt_mix': {
        'in_flan': False,
        'seq_len': {
            'max': 2,
            'mean': 1
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:ropes_read_background_situation': {
        'in_flan': False,
        'seq_len': {
            'max': 2,
            'mean': 1
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:rotten_tomatoes_Movie_Expressed_Sentiment': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice_separated_options'
    },
    't0_task_adaptation:rotten_tomatoes_Movie_Expressed_Sentiment_2': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice_separated_options'
    },
    't0_task_adaptation:rotten_tomatoes_Reviewer_Enjoyment': {
        'in_flan': True,
        'seq_len': {
            'max': 3,
            'mean': 3
        },
        'task_type': 't0_multiple_choice_separated_options'
    },
    't0_task_adaptation:rotten_tomatoes_Reviewer_Enjoyment_Yes_No': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice_separated_options'
    },
    't0_task_adaptation:rotten_tomatoes_Reviewer_Expressed_Sentiment': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice_separated_options'
    },
    't0_task_adaptation:rotten_tomatoes_Reviewer_Opinion_bad_good_choices': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:rotten_tomatoes_Reviewer_Sentiment_Feeling': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice_separated_options'
    },
    't0_task_adaptation:rotten_tomatoes_Sentiment_with_choices_': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:rotten_tomatoes_Text_Expressed_Sentiment': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice_separated_options'
    },
    't0_task_adaptation:rotten_tomatoes_Writer_Expressed_Sentiment': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice_separated_options'
    },
    't0_task_adaptation:samsum_Generate_a_summary_for_this_dialogue': {
        'in_flan': True,
        'seq_len': {
            'max': 57,
            'mean': 20
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:samsum_Given_the_above_dialogue_write_a_summary': {
        'in_flan': True,
        'seq_len': {
            'max': 57,
            'mean': 20
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:samsum_Sum_up_the_following_dialogue': {
        'in_flan': True,
        'seq_len': {
            'max': 57,
            'mean': 20
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:samsum_Summarize_': {
        'in_flan': True,
        'seq_len': {
            'max': 57,
            'mean': 20
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:samsum_Summarize_this_dialogue_': {
        'in_flan': True,
        'seq_len': {
            'max': 57,
            'mean': 20
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:samsum_To_sum_up_this_dialog': {
        'in_flan': True,
        'seq_len': {
            'max': 57,
            'mean': 20
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:samsum_Write_a_dialogue_that_match_this_summary': {
        'in_flan': True,
        'seq_len': {
            'max': 366,
            'mean': 93
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:sciq_Direct_Question': {
        'in_flan': False,
        'seq_len': {
            'max': 7,
            'mean': 1
        },
        'task_type': 't0_multiple_choice_separated_options'
    },
    't0_task_adaptation:sciq_Direct_Question_Closed_Book_': {
        'in_flan': False,
        'seq_len': {
            'max': 7,
            'mean': 1
        },
        'task_type': 't0_multiple_choice_separated_options'
    },
    't0_task_adaptation:sciq_Multiple_Choice': {
        'in_flan': False,
        'seq_len': {
            'max': 7,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:sciq_Multiple_Choice_Closed_Book_': {
        'in_flan': False,
        'seq_len': {
            'max': 7,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:sciq_Multiple_Choice_Question_First': {
        'in_flan': False,
        'seq_len': {
            'max': 7,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:social_i_qa_Check_if_a_random_answer_is_valid_or_not': {
        'in_flan': False,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice_separated_options'
    },
    't0_task_adaptation:social_i_qa_Generate_answer': {
        'in_flan': False,
        'seq_len': {
            'max': 10,
            'mean': 3
        },
        'task_type': 't0_multiple_choice_separated_options'
    },
    't0_task_adaptation:social_i_qa_Generate_the_question_from_the_answer': {
        'in_flan': False,
        'seq_len': {
            'max': 9,
            'mean': 6
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:social_i_qa_I_was_wondering': {
        'in_flan': False,
        'seq_len': {
            'max': 10,
            'mean': 3
        },
        'task_type': 't0_multiple_choice_separated_options'
    },
    't0_task_adaptation:social_i_qa_Show_choices_and_generate_answer': {
        'in_flan': False,
        'seq_len': {
            'max': 10,
            'mean': 3
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:social_i_qa_Show_choices_and_generate_index': {
        'in_flan': False,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:squad_v2_Jeopardy_with_Context': {
        'in_flan': True,
        'seq_len': {
            'max': 22,
            'mean': 10
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:squad_v2_Jeopardy_without_Context': {
        'in_flan': True,
        'seq_len': {
            'max': 22,
            'mean': 10
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:squad_v2_Questions_with_Context': {
        'in_flan': True,
        'seq_len': {
            'max': 6,
            'mean': 2
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:squad_v2_Questions_with_Context_Without_Prompt_Keywords':
        {
            'in_flan': True,
            'seq_len': {
                'max': 6,
                'mean': 2
            },
            'task_type': 't0_question_answer'
        },
    't0_task_adaptation:squad_v2_Questions_with_Context_Without_Prompt_Keywords_unanswerable':
        {
            'in_flan': True,
            'seq_len': {
                'max': 6,
                'mean': 2
            },
            'task_type': 't0_question_answer'
        },
    't0_task_adaptation:squad_v2_Questions_with_Context_unanswerable': {
        'in_flan': True,
        'seq_len': {
            'max': 6,
            'mean': 2
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:squad_v2_Topic_Prediction_Context': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:squad_v2_Topic_Prediction_Context_with_randomized_prompt_options':
        {
            'in_flan': True,
            'seq_len': {
                'max': 1,
                'mean': 1
            },
            'task_type': 't0_question_answer'
        },
    't0_task_adaptation:squad_v2_Topic_Prediction_Context_with_randomized_prompt_options_placed_in_the_end':
        {
            'in_flan': True,
            'seq_len': {
                'max': 1,
                'mean': 1
            },
            'task_type': 't0_question_answer'
        },
    't0_task_adaptation:squad_v2_Topic_Prediction_Question_and_Answer_Pair': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:squad_v2_Trivia': {
        'in_flan': True,
        'seq_len': {
            'max': 6,
            'mean': 2
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:squad_v2_Unanwerable_question': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice_separated_options'
    },
    't0_task_adaptation:super_glue_boolq_GPT_3_Style': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice_separated_options'
    },
    't0_task_adaptation:super_glue_boolq_I_wonder_': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice_separated_options'
    },
    't0_task_adaptation:super_glue_boolq_after_reading': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:super_glue_boolq_based_on_the_following_passage': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice_separated_options'
    },
    't0_task_adaptation:super_glue_boolq_based_on_the_previous_passage': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice_separated_options'
    },
    't0_task_adaptation:super_glue_boolq_could_you_tell_me_': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice_separated_options'
    },
    't0_task_adaptation:super_glue_boolq_exam': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:super_glue_boolq_exercise': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:super_glue_boolq_valid_binary': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:super_glue_boolq_yes_no_question': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:super_glue_cb_GPT_3_style': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:super_glue_cb_GPT_3_style_score_eval': {
        'in_flan': True,
        'task_type': 't0_multiple_choice_score_eval'
    },
    't0_task_adaptation:super_glue_cb_MNLI_crowdsource': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:super_glue_cb_MNLI_crowdsource_score_eval': {
        'in_flan': True,
        'task_type': 't0_multiple_choice_score_eval'
    },
    't0_task_adaptation:super_glue_cb_always_sometimes_never': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:super_glue_cb_always_sometimes_never_score_eval': {
        'in_flan': True,
        'task_type': 't0_multiple_choice_score_eval'
    },
    't0_task_adaptation:super_glue_cb_based_on_the_previous_passage': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:super_glue_cb_based_on_the_previous_passage_score_eval':
        {
            'in_flan': True,
            'task_type': 't0_multiple_choice_score_eval'
        },
    't0_task_adaptation:super_glue_cb_can_we_infer': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:super_glue_cb_can_we_infer_score_eval': {
        'in_flan': True,
        'task_type': 't0_multiple_choice_score_eval'
    },
    't0_task_adaptation:super_glue_cb_claim_true_false_inconclusive': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:super_glue_cb_claim_true_false_inconclusive_score_eval':
        {
            'in_flan': True,
            'task_type': 't0_multiple_choice_score_eval'
        },
    't0_task_adaptation:super_glue_cb_consider_always_sometimes_never': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:super_glue_cb_consider_always_sometimes_never_score_eval':
        {
            'in_flan': True,
            'task_type': 't0_multiple_choice_score_eval'
        },
    't0_task_adaptation:super_glue_cb_does_it_follow_that': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:super_glue_cb_does_it_follow_that_score_eval': {
        'in_flan': True,
        'task_type': 't0_multiple_choice_score_eval'
    },
    't0_task_adaptation:super_glue_cb_does_this_imply': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:super_glue_cb_does_this_imply_score_eval': {
        'in_flan': True,
        'task_type': 't0_multiple_choice_score_eval'
    },
    't0_task_adaptation:super_glue_cb_guaranteed_possible_impossible': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:super_glue_cb_guaranteed_possible_impossible_score_eval':
        {
            'in_flan': True,
            'task_type': 't0_multiple_choice_score_eval'
        },
    't0_task_adaptation:super_glue_cb_guaranteed_true': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:super_glue_cb_guaranteed_true_score_eval': {
        'in_flan': True,
        'task_type': 't0_multiple_choice_score_eval'
    },
    't0_task_adaptation:super_glue_cb_justified_in_saying': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:super_glue_cb_justified_in_saying_score_eval': {
        'in_flan': True,
        'task_type': 't0_multiple_choice_score_eval'
    },
    't0_task_adaptation:super_glue_cb_must_be_true': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:super_glue_cb_must_be_true_score_eval': {
        'in_flan': True,
        'task_type': 't0_multiple_choice_score_eval'
    },
    't0_task_adaptation:super_glue_cb_should_assume': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:super_glue_cb_should_assume_score_eval': {
        'in_flan': True,
        'task_type': 't0_multiple_choice_score_eval'
    },
    't0_task_adaptation:super_glue_cb_take_the_following_as_truth': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:super_glue_cb_take_the_following_as_truth_score_eval': {
        'in_flan': True,
        'task_type': 't0_multiple_choice_score_eval'
    },
    't0_task_adaptation:super_glue_copa_C1_or_C2_premise_so_because_': {
        'in_flan': True,
        'seq_len': {
            'max': 11,
            'mean': 5
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:super_glue_copa_C1_or_C2_premise_so_because__score_eval':
        {
            'in_flan': True,
            'task_type': 't0_multiple_choice_score_eval'
        },
    't0_task_adaptation:super_glue_copa__As_a_result_C1_or_C2_': {
        'in_flan': True,
        'seq_len': {
            'max': 10,
            'mean': 5
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:super_glue_copa__As_a_result_C1_or_C2__score_eval': {
        'in_flan': True,
        'task_type': 't0_multiple_choice_score_eval'
    },
    't0_task_adaptation:super_glue_copa__What_could_happen_next_C1_or_C2_': {
        'in_flan': True,
        'seq_len': {
            'max': 10,
            'mean': 5
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:super_glue_copa__What_could_happen_next_C1_or_C2__score_eval':
        {
            'in_flan': True,
            'task_type': 't0_multiple_choice_score_eval'
        },
    't0_task_adaptation:super_glue_copa__which_may_be_caused_by': {
        'in_flan': True,
        'seq_len': {
            'max': 11,
            'mean': 5
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:super_glue_copa__which_may_be_caused_by_score_eval': {
        'in_flan': True,
        'task_type': 't0_multiple_choice_score_eval'
    },
    't0_task_adaptation:super_glue_copa__why_C1_or_C2': {
        'in_flan': True,
        'seq_len': {
            'max': 11,
            'mean': 5
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:super_glue_copa__why_C1_or_C2_score_eval': {
        'in_flan': True,
        'task_type': 't0_multiple_choice_score_eval'
    },
    't0_task_adaptation:super_glue_copa_best_option': {
        'in_flan': True,
        'seq_len': {
            'max': 11,
            'mean': 5
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:super_glue_copa_best_option_score_eval': {
        'in_flan': True,
        'task_type': 't0_multiple_choice_score_eval'
    },
    't0_task_adaptation:super_glue_copa_cause_effect': {
        'in_flan': True,
        'seq_len': {
            'max': 11,
            'mean': 5
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:super_glue_copa_cause_effect_score_eval': {
        'in_flan': True,
        'task_type': 't0_multiple_choice_score_eval'
    },
    't0_task_adaptation:super_glue_copa_choose': {
        'in_flan': True,
        'seq_len': {
            'max': 11,
            'mean': 5
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:super_glue_copa_choose_score_eval': {
        'in_flan': True,
        'task_type': 't0_multiple_choice_score_eval'
    },
    't0_task_adaptation:super_glue_copa_exercise': {
        'in_flan': True,
        'seq_len': {
            'max': 11,
            'mean': 5
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:super_glue_copa_exercise_score_eval': {
        'in_flan': True,
        'task_type': 't0_multiple_choice_score_eval'
    },
    't0_task_adaptation:super_glue_copa_i_am_hesitating': {
        'in_flan': True,
        'seq_len': {
            'max': 11,
            'mean': 5
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:super_glue_copa_i_am_hesitating_score_eval': {
        'in_flan': True,
        'task_type': 't0_multiple_choice_score_eval'
    },
    't0_task_adaptation:super_glue_copa_more_likely': {
        'in_flan': True,
        'seq_len': {
            'max': 11,
            'mean': 5
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:super_glue_copa_more_likely_score_eval': {
        'in_flan': True,
        'task_type': 't0_multiple_choice_score_eval'
    },
    't0_task_adaptation:super_glue_copa_plausible_alternatives': {
        'in_flan': True,
        'seq_len': {
            'max': 11,
            'mean': 5
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:super_glue_copa_plausible_alternatives_score_eval': {
        'in_flan': True,
        'task_type': 't0_multiple_choice_score_eval'
    },
    't0_task_adaptation:super_glue_multirc_I_was_going_to_say_': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice_separated_options'
    },
    't0_task_adaptation:super_glue_multirc_Would_it_be_good_to_answer_': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice_separated_options'
    },
    't0_task_adaptation:super_glue_multirc_confirm': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:super_glue_multirc_correct': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice_separated_options'
    },
    't0_task_adaptation:super_glue_multirc_decide_valid': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:super_glue_multirc_found_this_answer': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:super_glue_multirc_grading': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice_separated_options'
    },
    't0_task_adaptation:super_glue_multirc_is_a_correct_answer_': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice_separated_options'
    },
    't0_task_adaptation:super_glue_multirc_is_the_correct_answer_': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice_separated_options'
    },
    't0_task_adaptation:super_glue_multirc_paragraph_question_is_it_': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice_separated_options'
    },
    't0_task_adaptation:super_glue_record_Add_sentence_after_after_continuation_choices_':
        {
            'in_flan': True,
            'seq_len': {
                'max': 52,
                'mean': 22
            },
            'task_type': 't0_multiple_choice_separated_options'
        },
    't0_task_adaptation:super_glue_record_Add_sentence_after_continuation_choices_':
        {
            'in_flan': True,
            'seq_len': {
                'max': 52,
                'mean': 22
            },
            'task_type': 't0_multiple_choice_separated_options'
        },
    't0_task_adaptation:super_glue_record_Can_you_figure_out_': {
        'in_flan': True,
        'seq_len': {
            'max': 4,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:super_glue_record_GPT_3_style_continuation_choices_': {
        'in_flan': True,
        'seq_len': {
            'max': 53,
            'mean': 23
        },
        'task_type': 't0_multiple_choice_separated_options'
    },
    't0_task_adaptation:super_glue_record_GPT_3_style_summary_only_continuation_choices_':
        {
            'in_flan': True,
            'seq_len': {
                'max': 53,
                'mean': 23
            },
            'task_type': 't0_multiple_choice_separated_options'
        },
    't0_task_adaptation:super_glue_record_GPT_3_style_with_labels_continuation_choices_':
        {
            'in_flan': True,
            'seq_len': {
                'max': 53,
                'mean': 23
            },
            'task_type': 't0_multiple_choice_separated_options'
        },
    't0_task_adaptation:super_glue_record_GPT_3_style_with_labels_without_hyphens_continuation_choices_':
        {
            'in_flan': True,
            'seq_len': {
                'max': 52,
                'mean': 22
            },
            'task_type': 't0_multiple_choice_separated_options'
        },
    't0_task_adaptation:super_glue_record_GPT_3_style_without_hyphens_continuation_choices_':
        {
            'in_flan': True,
            'seq_len': {
                'max': 52,
                'mean': 22
            },
            'task_type': 't0_multiple_choice_separated_options'
        },
    't0_task_adaptation:super_glue_record_In_the_question_above_the_placeholder_stands_for':
        {
            'in_flan': True,
            'seq_len': {
                'max': 4,
                'mean': 1
            },
            'task_type': 't0_multiple_choice'
        },
    't0_task_adaptation:super_glue_record_New_highlight_continuation_choices_':
        {
            'in_flan': True,
            'seq_len': {
                'max': 53,
                'mean': 23
            },
            'task_type': 't0_multiple_choice_separated_options'
        },
    't0_task_adaptation:super_glue_record_News_article_continuation_choices_': {
        'in_flan': True,
        'seq_len': {
            'max': 52,
            'mean': 22
        },
        'task_type': 't0_multiple_choice_separated_options'
    },
    't0_task_adaptation:super_glue_record_Summary_first_continuation_choices_':
        {
            'in_flan': True,
            'seq_len': {
                'max': 52,
                'mean': 22
            },
            'task_type': 't0_multiple_choice_separated_options'
        },
    't0_task_adaptation:super_glue_record_What_could_the_placeholder_be_': {
        'in_flan': True,
        'seq_len': {
            'max': 4,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:super_glue_record_Which_one_is_the_placeholder_': {
        'in_flan': True,
        'seq_len': {
            'max': 4,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:super_glue_record_choose_between': {
        'in_flan': True,
        'seq_len': {
            'max': 4,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:super_glue_record_corrupted': {
        'in_flan': True,
        'seq_len': {
            'max': 4,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:super_glue_record_exercise': {
        'in_flan': True,
        'seq_len': {
            'max': 4,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:super_glue_record_pick_one_option': {
        'in_flan': True,
        'seq_len': {
            'max': 4,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:super_glue_record_the_placeholder_refers_to_': {
        'in_flan': True,
        'seq_len': {
            'max': 4,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:super_glue_record_trying_to_decide': {
        'in_flan': True,
        'seq_len': {
            'max': 4,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:super_glue_rte_GPT_3_style': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:super_glue_rte_GPT_3_style_score_eval': {
        'in_flan': True,
        'task_type': 't0_multiple_choice_score_eval'
    },
    't0_task_adaptation:super_glue_rte_MNLI_crowdsource': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:super_glue_rte_MNLI_crowdsource_score_eval': {
        'in_flan': True,
        'task_type': 't0_multiple_choice_score_eval'
    },
    't0_task_adaptation:super_glue_rte_based_on_the_previous_passage': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:super_glue_rte_based_on_the_previous_passage_score_eval':
        {
            'in_flan': True,
            'task_type': 't0_multiple_choice_score_eval'
        },
    't0_task_adaptation:super_glue_rte_can_we_infer': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:super_glue_rte_can_we_infer_score_eval': {
        'in_flan': True,
        'task_type': 't0_multiple_choice_score_eval'
    },
    't0_task_adaptation:super_glue_rte_does_it_follow_that': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:super_glue_rte_does_it_follow_that_score_eval': {
        'in_flan': True,
        'task_type': 't0_multiple_choice_score_eval'
    },
    't0_task_adaptation:super_glue_rte_does_this_imply': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:super_glue_rte_does_this_imply_score_eval': {
        'in_flan': True,
        'task_type': 't0_multiple_choice_score_eval'
    },
    't0_task_adaptation:super_glue_rte_guaranteed_true': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:super_glue_rte_guaranteed_true_score_eval': {
        'in_flan': True,
        'task_type': 't0_multiple_choice_score_eval'
    },
    't0_task_adaptation:super_glue_rte_justified_in_saying': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:super_glue_rte_justified_in_saying_score_eval': {
        'in_flan': True,
        'task_type': 't0_multiple_choice_score_eval'
    },
    't0_task_adaptation:super_glue_rte_must_be_true': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:super_glue_rte_must_be_true_score_eval': {
        'in_flan': True,
        'task_type': 't0_multiple_choice_score_eval'
    },
    't0_task_adaptation:super_glue_rte_should_assume': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:super_glue_rte_should_assume_score_eval': {
        'in_flan': True,
        'task_type': 't0_multiple_choice_score_eval'
    },
    't0_task_adaptation:super_glue_wic_GPT_3_prompt': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice_separated_options'
    },
    't0_task_adaptation:super_glue_wic_GPT_3_prompt_score_eval': {
        'in_flan': True,
        'task_type': 't0_multiple_choice_score_eval'
    },
    't0_task_adaptation:super_glue_wic_GPT_3_prompt_with_label': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:super_glue_wic_GPT_3_prompt_with_label_score_eval': {
        'in_flan': True,
        'task_type': 't0_multiple_choice_score_eval'
    },
    't0_task_adaptation:super_glue_wic_affirmation_true_or_false': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:super_glue_wic_affirmation_true_or_false_score_eval': {
        'in_flan': True,
        'task_type': 't0_multiple_choice_score_eval'
    },
    't0_task_adaptation:super_glue_wic_grammar_homework': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:super_glue_wic_grammar_homework_score_eval': {
        'in_flan': True,
        'task_type': 't0_multiple_choice_score_eval'
    },
    't0_task_adaptation:super_glue_wic_polysemous': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:super_glue_wic_polysemous_score_eval': {
        'in_flan': True,
        'task_type': 't0_multiple_choice_score_eval'
    },
    't0_task_adaptation:super_glue_wic_question_context': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice_separated_options'
    },
    't0_task_adaptation:super_glue_wic_question_context_meaning': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice_separated_options'
    },
    't0_task_adaptation:super_glue_wic_question_context_meaning_score_eval': {
        'in_flan': True,
        'task_type': 't0_multiple_choice_score_eval'
    },
    't0_task_adaptation:super_glue_wic_question_context_meaning_with_label': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:super_glue_wic_question_context_meaning_with_label_score_eval':
        {
            'in_flan': True,
            'task_type': 't0_multiple_choice_score_eval'
        },
    't0_task_adaptation:super_glue_wic_question_context_score_eval': {
        'in_flan': True,
        'task_type': 't0_multiple_choice_score_eval'
    },
    't0_task_adaptation:super_glue_wic_same_sense': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:super_glue_wic_same_sense_score_eval': {
        'in_flan': True,
        'task_type': 't0_multiple_choice_score_eval'
    },
    't0_task_adaptation:super_glue_wic_similar_sense': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice_separated_options'
    },
    't0_task_adaptation:super_glue_wic_similar_sense_score_eval': {
        'in_flan': True,
        'task_type': 't0_multiple_choice_score_eval'
    },
    't0_task_adaptation:super_glue_wsc.fixed_GPT_3_Style': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice_separated_options'
    },
    't0_task_adaptation:super_glue_wsc.fixed_GPT_3_Style_score_eval': {
        'in_flan': True,
        'task_type': 't0_multiple_choice_score_eval'
    },
    't0_task_adaptation:super_glue_wsc.fixed_I_think_they_mean': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:super_glue_wsc.fixed_I_think_they_mean_score_eval': {
        'in_flan': True,
        'task_type': 't0_multiple_choice_score_eval'
    },
    't0_task_adaptation:super_glue_wsc.fixed_Who_or_what_is_are': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice_separated_options'
    },
    't0_task_adaptation:super_glue_wsc.fixed_Who_or_what_is_are_score_eval': {
        'in_flan': True,
        'task_type': 't0_multiple_choice_score_eval'
    },
    't0_task_adaptation:super_glue_wsc.fixed_by_p_they_mean': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:super_glue_wsc.fixed_by_p_they_mean_score_eval': {
        'in_flan': True,
        'task_type': 't0_multiple_choice_score_eval'
    },
    't0_task_adaptation:super_glue_wsc.fixed_does_p_stand_for': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:super_glue_wsc.fixed_does_p_stand_for_score_eval': {
        'in_flan': True,
        'task_type': 't0_multiple_choice_score_eval'
    },
    't0_task_adaptation:super_glue_wsc.fixed_does_the_pronoun_refer_to': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:super_glue_wsc.fixed_does_the_pronoun_refer_to_score_eval':
        {
            'in_flan': True,
            'task_type': 't0_multiple_choice_score_eval'
        },
    't0_task_adaptation:super_glue_wsc.fixed_in_other_words': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:super_glue_wsc.fixed_in_other_words_score_eval': {
        'in_flan': True,
        'task_type': 't0_multiple_choice_score_eval'
    },
    't0_task_adaptation:super_glue_wsc.fixed_p_is_are_r': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:super_glue_wsc.fixed_p_is_are_r_score_eval': {
        'in_flan': True,
        'task_type': 't0_multiple_choice_score_eval'
    },
    't0_task_adaptation:super_glue_wsc.fixed_replaced_with': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:super_glue_wsc.fixed_replaced_with_score_eval': {
        'in_flan': True,
        'task_type': 't0_multiple_choice_score_eval'
    },
    't0_task_adaptation:super_glue_wsc.fixed_the_pronoun_refers_to': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:super_glue_wsc.fixed_the_pronoun_refers_to_score_eval':
        {
            'in_flan': True,
            'task_type': 't0_multiple_choice_score_eval'
        },
    't0_task_adaptation:trec_fine_grained_ABBR': {
        'in_flan': True,
        'seq_len': {
            'max': 2,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:trec_fine_grained_ABBR_context_first': {
        'in_flan': True,
        'seq_len': {
            'max': 2,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:trec_fine_grained_DESC': {
        'in_flan': True,
        'seq_len': {
            'max': 3,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:trec_fine_grained_DESC_context_first': {
        'in_flan': True,
        'seq_len': {
            'max': 3,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:trec_fine_grained_ENTY': {
        'in_flan': True,
        'seq_len': {
            'max': 5,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:trec_fine_grained_HUM': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:trec_fine_grained_HUM_context_first': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:trec_fine_grained_LOC': {
        'in_flan': True,
        'seq_len': {
            'max': 2,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:trec_fine_grained_LOC_context_first': {
        'in_flan': True,
        'seq_len': {
            'max': 2,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:trec_fine_grained_NUM': {
        'in_flan': True,
        'seq_len': {
            'max': 3,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:trec_fine_grained_NUM_context_first': {
        'in_flan': True,
        'seq_len': {
            'max': 3,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:trec_fine_grained_open': {
        'in_flan': True,
        'seq_len': {
            'max': 4,
            'mean': 1
        },
        'task_type': 't0_multiple_choice_separated_options'
    },
    't0_task_adaptation:trec_fine_grained_open_context_first': {
        'in_flan': True,
        'seq_len': {
            'max': 4,
            'mean': 1
        },
        'task_type': 't0_multiple_choice_separated_options'
    },
    't0_task_adaptation:trec_pick_the_best_descriptor': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:trec_trec1': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:trec_trec2': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:trec_what_category_best_describe': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:trec_which_category_best_describes': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:trivia_qa_unfiltered_first_person_context': {
        'in_flan': True,
        'seq_len': {
            'max': 43,
            'mean': 4
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:trivia_qa_unfiltered_formal_description': {
        'in_flan': True,
        'seq_len': {
            'max': 43,
            'mean': 4
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:trivia_qa_unfiltered_guess_question': {
        'in_flan': True,
        'seq_len': {
            'max': 35,
            'mean': 11
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:trivia_qa_unfiltered_question_answer': {
        'in_flan': True,
        'seq_len': {
            'max': 43,
            'mean': 4
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:trivia_qa_unfiltered_question_with_instruction': {
        'in_flan': True,
        'seq_len': {
            'max': 43,
            'mean': 4
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:web_questions_get_the_answer': {
        'in_flan': False,
        'seq_len': {
            'max': 10,
            'mean': 2
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:web_questions_potential_correct_answer': {
        'in_flan': False,
        'seq_len': {
            'max': 10,
            'mean': 2
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:web_questions_question_answer': {
        'in_flan': False,
        'seq_len': {
            'max': 10,
            'mean': 2
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:web_questions_short_general_knowledge_q': {
        'in_flan': False,
        'seq_len': {
            'max': 10,
            'mean': 2
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:web_questions_whats_the_answer': {
        'in_flan': False,
        'seq_len': {
            'max': 10,
            'mean': 2
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:wiki_bio_comprehension': {
        'in_flan': False,
        'seq_len': {
            'max': 273,
            'mean': 72
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:wiki_bio_guess_person': {
        'in_flan': False,
        'seq_len': {
            'max': 9,
            'mean': 2
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:wiki_bio_key_content': {
        'in_flan': False,
        'seq_len': {
            'max': 273,
            'mean': 72
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:wiki_bio_what_content': {
        'in_flan': False,
        'seq_len': {
            'max': 84,
            'mean': 26
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:wiki_bio_who': {
        'in_flan': False,
        'seq_len': {
            'max': 392,
            'mean': 92
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:wiki_hop_original_choose_best_object_affirmative_1': {
        'in_flan': False,
        'seq_len': {
            'max': 7,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:wiki_hop_original_choose_best_object_affirmative_2': {
        'in_flan': False,
        'seq_len': {
            'max': 7,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:wiki_hop_original_choose_best_object_affirmative_3': {
        'in_flan': False,
        'seq_len': {
            'max': 7,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:wiki_hop_original_choose_best_object_interrogative_1': {
        'in_flan': False,
        'seq_len': {
            'max': 7,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:wiki_hop_original_choose_best_object_interrogative_2': {
        'in_flan': False,
        'seq_len': {
            'max': 7,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:wiki_hop_original_explain_relation': {
        'in_flan': False,
        'seq_len': {
            'max': 6,
            'mean': 2
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:wiki_hop_original_generate_object': {
        'in_flan': False,
        'seq_len': {
            'max': 7,
            'mean': 1
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:wiki_hop_original_generate_subject': {
        'in_flan': False,
        'seq_len': {
            'max': 10,
            'mean': 2
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:wiki_hop_original_generate_subject_and_object': {
        'in_flan': False,
        'seq_len': {
            'max': 12,
            'mean': 4
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:wiki_qa_Decide_good_answer': {
        'in_flan': False,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:wiki_qa_Direct_Answer_to_Question': {
        'in_flan': False,
        'seq_len': {
            'max': 69,
            'mean': 26
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:wiki_qa_Generate_Question_from_Topic': {
        'in_flan': False,
        'seq_len': {
            'max': 19,
            'mean': 6
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:wiki_qa_Is_This_True_': {
        'in_flan': False,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice_separated_options'
    },
    't0_task_adaptation:wiki_qa_Jeopardy_style': {
        'in_flan': False,
        'seq_len': {
            'max': 19,
            'mean': 6
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:wiki_qa_Topic_Prediction_Answer_Only': {
        'in_flan': False,
        'seq_len': {
            'max': 12,
            'mean': 2
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:wiki_qa_Topic_Prediction_Question_Only': {
        'in_flan': False,
        'seq_len': {
            'max': 12,
            'mean': 2
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:wiki_qa_Topic_Prediction_Question_and_Answer_Pair': {
        'in_flan': False,
        'seq_len': {
            'max': 12,
            'mean': 2
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:wiki_qa_automatic_system': {
        'in_flan': False,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice_separated_options'
    },
    't0_task_adaptation:wiki_qa_exercise': {
        'in_flan': False,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:wiki_qa_found_on_google': {
        'in_flan': False,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:winogrande_winogrande_debiased_Replace': {
        'in_flan': True,
        'seq_len': {
            'max': 3,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:winogrande_winogrande_debiased_Replace_score_eval': {
        'in_flan': True,
        'task_type': 't0_multiple_choice_score_eval'
    },
    't0_task_adaptation:winogrande_winogrande_debiased_does_underscore_refer_to':
        {
            'in_flan': True,
            'seq_len': {
                'max': 3,
                'mean': 1
            },
            'task_type': 't0_multiple_choice'
        },
    't0_task_adaptation:winogrande_winogrande_debiased_does_underscore_refer_to_score_eval':
        {
            'in_flan': True,
            'task_type': 't0_multiple_choice_score_eval'
        },
    't0_task_adaptation:winogrande_winogrande_debiased_fill_in_the_blank': {
        'in_flan': True,
        'seq_len': {
            'max': 3,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:winogrande_winogrande_debiased_fill_in_the_blank_score_eval':
        {
            'in_flan': True,
            'task_type': 't0_multiple_choice_score_eval'
        },
    't0_task_adaptation:winogrande_winogrande_debiased_stand_for': {
        'in_flan': True,
        'seq_len': {
            'max': 3,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:winogrande_winogrande_debiased_stand_for_score_eval': {
        'in_flan': True,
        'task_type': 't0_multiple_choice_score_eval'
    },
    't0_task_adaptation:winogrande_winogrande_debiased_underscore_refer_to': {
        'in_flan': True,
        'seq_len': {
            'max': 3,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:winogrande_winogrande_debiased_underscore_refer_to_score_eval':
        {
            'in_flan': True,
            'task_type': 't0_multiple_choice_score_eval'
        },
    't0_task_adaptation:winogrande_winogrande_xl_Replace': {
        'in_flan': True,
        'seq_len': {
            'max': 2,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:winogrande_winogrande_xl_Replace_score_eval': {
        'in_flan': True,
        'task_type': 't0_multiple_choice_score_eval'
    },
    't0_task_adaptation:winogrande_winogrande_xl_does_underscore_refer_to': {
        'in_flan': True,
        'seq_len': {
            'max': 2,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:winogrande_winogrande_xl_does_underscore_refer_to_score_eval':
        {
            'in_flan': True,
            'task_type': 't0_multiple_choice_score_eval'
        },
    't0_task_adaptation:winogrande_winogrande_xl_fill_in_the_blank': {
        'in_flan': True,
        'seq_len': {
            'max': 2,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:winogrande_winogrande_xl_fill_in_the_blank_score_eval':
        {
            'in_flan': True,
            'task_type': 't0_multiple_choice_score_eval'
        },
    't0_task_adaptation:winogrande_winogrande_xl_stand_for': {
        'in_flan': True,
        'seq_len': {
            'max': 2,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:winogrande_winogrande_xl_stand_for_score_eval': {
        'in_flan': True,
        'task_type': 't0_multiple_choice_score_eval'
    },
    't0_task_adaptation:winogrande_winogrande_xl_underscore_refer_to': {
        'in_flan': True,
        'seq_len': {
            'max': 2,
            'mean': 1
        },
        'task_type': 't0_multiple_choice'
    },
    't0_task_adaptation:winogrande_winogrande_xl_underscore_refer_to_score_eval':
        {
            'in_flan': True,
            'task_type': 't0_multiple_choice_score_eval'
        },
    't0_task_adaptation:wiqa_does_the_supposed_perturbation_have_an_effect': {
        'in_flan': False,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:wiqa_effect_with_label_answer': {
        'in_flan': False,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:wiqa_effect_with_string_answer': {
        'in_flan': False,
        'seq_len': {
            'max': 2,
            'mean': 1
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:wiqa_what_is_the_final_step_of_the_following_process': {
        'in_flan': False,
        'seq_len': {
            'max': 26,
            'mean': 8
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:wiqa_what_is_the_missing_first_step': {
        'in_flan': False,
        'seq_len': {
            'max': 21,
            'mean': 7
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:wiqa_what_might_be_the_first_step_of_the_process': {
        'in_flan': False,
        'seq_len': {
            'max': 21,
            'mean': 7
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:wiqa_what_might_be_the_last_step_of_the_process': {
        'in_flan': False,
        'seq_len': {
            'max': 26,
            'mean': 8
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:wiqa_which_of_the_following_is_the_supposed_perturbation':
        {
            'in_flan': False,
            'seq_len': {
                'max': 7,
                'mean': 7
            },
            'task_type': 't0_question_answer'
        },
    't0_task_adaptation:xsum_DOC_boils_down_to_simple_idea_that': {
        'in_flan': True,
        'seq_len': {
            'max': 38,
            'mean': 20
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:xsum_DOC_given_above_write_one_sentence': {
        'in_flan': True,
        'seq_len': {
            'max': 38,
            'mean': 20
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:xsum_DOC_how_would_you_rephrase_few_words': {
        'in_flan': True,
        'seq_len': {
            'max': 38,
            'mean': 20
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:xsum_DOC_tldr': {
        'in_flan': True,
        'seq_len': {
            'max': 38,
            'mean': 20
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:xsum_DOC_write_summary_of_above': {
        'in_flan': True,
        'seq_len': {
            'max': 38,
            'mean': 20
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:xsum_article_DOC_summary': {
        'in_flan': True,
        'seq_len': {
            'max': 38,
            'mean': 20
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:xsum_college_roommate_asked_DOC_so_I_recap': {
        'in_flan': True,
        'seq_len': {
            'max': 38,
            'mean': 20
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:xsum_read_below_DOC_write_abstract': {
        'in_flan': True,
        'seq_len': {
            'max': 38,
            'mean': 20
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:xsum_summarize_DOC': {
        'in_flan': True,
        'seq_len': {
            'max': 38,
            'mean': 20
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:xsum_summarize_this_DOC_summary': {
        'in_flan': True,
        'seq_len': {
            'max': 38,
            'mean': 20
        },
        'task_type': 't0_question_answer'
    },
    't0_task_adaptation:yelp_review_full_based_on_that': {
        'in_flan': True,
        'seq_len': {
            'max': 2,
            'mean': 2
        },
        'task_type': 't0_multiple_choice_separated_options'
    },
    't0_task_adaptation:yelp_review_full_format_rating': {
        'in_flan': True,
        'seq_len': {
            'max': 2,
            'mean': 2
        },
        'task_type': 't0_multiple_choice_separated_options'
    },
    't0_task_adaptation:yelp_review_full_format_score': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice_separated_options'
    },
    't0_task_adaptation:yelp_review_full_format_star': {
        'in_flan': True,
        'seq_len': {
            'max': 2,
            'mean': 2
        },
        'task_type': 't0_multiple_choice_separated_options'
    },
    't0_task_adaptation:yelp_review_full_on_a_scale': {
        'in_flan': True,
        'seq_len': {
            'max': 1,
            'mean': 1
        },
        'task_type': 't0_multiple_choice_separated_options'
    },
    't0_task_adaptation:yelp_review_full_so_i_would': {
        'in_flan': True,
        'seq_len': {
            'max': 2,
            'mean': 2
        },
        'task_type': 't0_multiple_choice_separated_options'
    },
    't0_task_adaptation:yelp_review_full_this_place': {
        'in_flan': True,
        'seq_len': {
            'max': 2,
            'mean': 2
        },
        'task_type': 't0_multiple_choice_separated_options'
    }
}

T0_TRAIN_TASKS_ABBREV = [
    k.replace('t0_task_adaptation:', '') for k in T0_TRAIN_TASK_METADATA.keys()
]
