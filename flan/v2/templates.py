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

"""Templates/Prompts."""
import copy
import dataclasses

from flan.v2 import flan_templates_branched as flan_templates

# pylint: disable=implicit-str-concat

# Zero-shot patterns.
PATTERNS = {
    "rte": [
        ("{premise}\n\nQuestion with options: Based on the paragraph above can"
         " we conclude that \"{hypothesis}\"?\n\n{options_}", "{answer}"),
        ("{premise}\n\nBased on that paragraph can we conclude that the "
         "sentence below is true?\n{hypothesis}\n\n{options_}", "{answer}"),
        ("{premise}\n\nQ with options: Can we draw the following "
         "conclusion?\n{hypothesis}\n\n{options_}", "{answer}"),
        ("{premise}\nDoes this next sentence follow, given the preceding "
         "text?\n{hypothesis}\n\n{options_}", "{answer}"),
        ("{premise}\n{options_}\nQuestion: Can we infer the "
         "following?\n{hypothesis}", "{answer}"),
        ("Read the following paragraph and determine if the hypothesis is "
         "true. Select from options at the end:\n\n{premise}\n\nHypothesis: "
         "{hypothesis}\n{options_}\nThe answer is", "{answer}"),
        ("Read the text and determine if the sentence is "
         "true:\n\n{premise}\n\nSentence: {hypothesis}\n{options_}\nA:",
         "{answer}"),
        ("Question with options: can we draw the following hypothesis from the"
         " context? \n\nContext:\n\n{premise}\n\nHypothesis: "
         "{hypothesis}\n{options_}\nA:", "{answer}"),
        ("Determine if the sentence is true based on the text below. Choose "
         "from options.\n{hypothesis}\n\n{premise}\n{options_}", "{answer}"),
        ("Generate a context and a hypothesis.",
         "Context: {premise}\n\nHypothesis: {hypothesis}"),
    ],
    "wsc": [
        ("{context}\n\nWhich option(s) below is/are correct for question: are "
         "\"{text1}\" and \"{text2}\" the same entity?\n\n{options_}",
         "{answer}"),
        ("{context}\n\nMulti-choice question: Do \"{text1}\" and \"{text2}\" "
         "have the same meaning?\n\n{options_}", "{answer}"),
        ("Multi-choice problem: Given the following "
         "context\n\n{context}\n\nAre \"{text1}\" and \"{text2}\" the "
         "same?\n\n{options_}\nA:", "{answer}"),
        ("Choose your answer.{options_}.\n\n{context}\n\nDo \"{text2}\" and "
         "\"{text1}\" mean the same thing?", "{answer}"),
        ("{context}\n\nAre \"{text2}\" and \"{text1}\" the same thing in the "
         "aforementioned sentence (choose from options)?\n\n{options_}",
         "{answer}"),
        ("Context:{context}\n\nIs \"{text2}\" the same as \"{text1}\"? "
         "Possible answers:{options_}\n\nAnswer:", "{answer}"),
        ("Consider this sentence: {context}\n\nAre \"{text2}\" and \"{text1}\""
         " the same (see options)?\n\n{options_}", "{answer}"),
        ("Are \"{text1}\" and \"{text2}\" the same in this "
         "sentence?\n{context}\n\n{options_}\nThe answer is:", "{answer}"),
        ("See context followed by options. Is \"{text1}\" the same as "
         "\"{text2}\" in this sentence?\n{context}\n\n{options_}", "{answer}"),
        ("Choose your answer: Do \"{text1}\" and \"{text2}\" point to the same"
         " thing in the following sentence?\n\n{context}\n\n{options_}",
         "{answer}"),
    ],
    "wsc273": [
        ("Multi-choice problem: {context}\n{options_}", "{answer}"),
        ("Complete the passage.\n\n{context}\n{options_}", "{answer}"),
        ("How does this following sentence end (see "
         "options)?\n\n{context}\n{options_}", "{answer}"),
        ("What is the most logical completion for the following text (see "
         "options)?\n\n{context}\n{options_}", "{answer}"),
        ("Multi-choice problem: How does this text "
         "end?\n\n{context}\n{options_}", "{answer}"),
        ("Choose from the options on what happens "
         "next.\n\n{context}\n{options_}", "{answer}"),
        ("Complete the following sentence.\n\n{context}\n{options_}",
         "{answer}"),
        ("Choose from options: Fill in the remainder of the "
         "sentence.\n\n{context}\n{options_}", "{answer}"),
        ("What is the next event listed in the options is "
         "correct?\n\n{context}\n{options_}\nA:", "{answer}"),
        ("Complete the rest of the sentence by choosing from "
         "options.\n\n{context}\n{options_}", "{answer}"),
    ],
    "wic": [
        ("{sentence1}\n{sentence2}\nChoose your answer: Does the word "
         "\"{word}\" mean the same thing in the above two "
         "sentences?\n{options_}", "{answer}"),
        ("Sentence 1: {sentence1}\nSentence 2: {sentence2}\nMulti-choice "
         "problem: Does {word} mean the same thing in these two "
         "sentences?\n{options_}", "{answer}"),
        ("Here is one sentence: {sentence1}\nHere is another sentence: "
         "{sentence2}\nQ: Does the term {word} mean the same thing in both "
         "these sentences?\n{options_}", "{answer}"),
        ("In these two sentences (1) {sentence1} (2) {sentence2}, does the "
         "word {word} mean the same thing?\n{options_}.... A:", "{answer}"),
        ("Multi-choice problem: does word \"{word}\" have the same meaning in "
         "the following two "
         "sentences?\n\n{sentence1}\n\n{sentence2}\n\n{options_}", "{answer}"),
        ("This question has options. Is the word \"{word}\" used in the same "
         "way in the following two "
         "sentences?\n\n{sentence1}\n\n{sentence2}\n\n{options_}", "{answer}"),
        ("This question has options. Does the word \"{word}\" have the same "
         "definition in the next two "
         "sentences?\n\n{sentence1}\n\n{sentence2}\n\n{options_}", "{answer}"),
        ("Is {word} used to mean the same thing in the next two sentences (see"
         " options)?\n\n{sentence1}\n\n{sentence2}\n\n{options_}", "{answer}"),
        ("Does \"{word}\" mean the same thing in these two sentences? See "
         "options at the end. \n{sentence1}\n{sentence2}\n{options_}..Answer:",
         "{answer}"),
        ("(options at the end). Does the word \"{word}\" mean the same thing "
         "in \"{sentence1}\" and \"{sentence2}\"?\n{options_}", "{answer}"),
    ],
    "record": [
        ("Complete the passage: pick from possible "
         "candidates.\n\n{passage}\n\n{query}\n\n{options_str}\n\n",
         "{answer}"),
        ("{passage}\n\n{query}\n\n{options_str}\n\n", "{answer}"),
        ("Find the right ending to this "
         "passage.\n\n{passage}\n\n{query}\n\n{options_str}\n\n", "{answer}"),
        ("What's the most logical way to complete this "
         "passage?\n\n{passage}\n\n{query}\n\n{options_str}\n\n", "{answer}"),
        ("Choose the next sentence."
         "{passage}\n\n{query}\n\n{options_str}\n\n", "{answer}"),
        ("Choose how you want this story to "
         "end.\n\n{passage}\n\n{query}\n\n{options_str}\n\n", "{answer}"),
        ("Write the last sentence in this "
         "story.\n\n{passage}\n\n{query}\n\n{options_str}\n\n", "{answer}"),
        ("Choose the next sentence for this "
         "paragraph.\n\n{passage}\n\n{query}\n\n{options_str}\n\n", "{answer}"),
        ("What is the most logical completion of this news "
         "story?.\n\n{passage}\n\n{query}\n\n{options_str}\n\n", "{answer}"),
        ("How does the sentence end?\n\n"
         "{passage}\n\n{query}\n\n{options_str}\n\n", "{answer}"),
    ],
    "natural_questions": [
        ("Question: {question}?\nAnswer:", "{answer}"),
        ("{question}?", "{answer}"),
        ("Answer the following question:\n\n{question}", "{answer}"),
        ("Answer this question:\n\n{question}?", "{answer}"),
        ("Please answer this question: {question}", "{answer}"),
        ("Answer the question...{question}?", "{answer}"),
        ("What is the answer to this question? {question}\n\n", "{answer}"),
        ("Can you tell me the answer to {question}?", "{answer}"),
        ("Next question: {question}\n\n", "{answer}"),
        ("Q: {question} A:", "{answer}"),
    ],
    # Not in FLAN Templates (flan_templates):
    "synth_cot_natural_questions": [
        ("Question: {question}?\nCoT:", "{cot}\nThe answer is {answer}"),
        ("{question}? Let's think step by step.",
         "{cot}\n\nThe answer is {answer}"),
        ("Answer the following question carefully:\n\n{question}",
         "\n{cot}\nThe answer is {answer}"),
        ("Answer this question:\n\n{question}? Think out loud!",
         "{cot}\nSo, the answer is {answer}"),
        ("Please answer this question: {question}\nGive your reasons first.",
         "{cot}\n\nThe answer: {answer}"),
        ("Answer the question...{question}? Give your explanation afterwards",
         "The answer: {answer}\nExplanation: {cot}"),
        ("What is the answer to this question? {question}\nLet's think...",
         "{cot}. So the answer is {answer}."),
        ("Can you tell me the logic and answer to {question}?",
         "logic: {cot}\n\nThe final answer: {answer}"),
        ("Next question: {question}\n\nSolution:",
         "{cot}\nThe answer is {answer}"),
        ("Q: {question} Step-by-step reasoning process:",
         "{cot} The answer is {answer}"),
    ],
    "trivia_qa": [
        ("Please answer this question: {question}", "{answer}"),
        ("{question}", "{answer}"),
        ("Write the answer: {question}", "{answer}"),
        ("What is the answer: {question}", "{answer}"),
        ("Answer this question.\n\n{question}", "{answer}"),
        ("Answer the following question. {question}", "{answer}"),
        ("Question: {question}\nAnswer:", "{answer}"),
        ("{question}???", "{answer}"),
        ("Trivia question: {question}\nAnd the answer is?", "{answer}"),
        ("{question}\nWhat is the answer?", "{answer}"),
    ],
    "math_dataset": [
        ("{question}", "{answer}"),
        ("Solve this math problem\n\n{question}", "{answer}"),
        ("What is the solution?\n\n{question}", "{answer}"),
        ("Math Problem\n{question}", "{answer}"),
        ("Write down the solution for this math problem: {question}",
         "{answer}"),
        ("What is the solution to this math problem?\n{question}", "{answer}"),
        ("Math problem: {question}\nWhat is the solution?", "{answer}"),
        ("{question}\nSolve this problem.", "{answer}"),
        ("Problem: {question}\nAnd the answer is...", "{answer}"),
        ("{question}. What is the answer??", "{answer}"),
    ],
    "aeslc": [
        ("What is the subject line for this email?\n\n{body}\n\nSubject Line:",
         "{subject}"),
        ("Write a subject line for this message:\n\n{body}\n\nSubject Line:",
         "{subject}"),
        ("{body}\nWrite a subject line for this email.", "{subject}"),
        ("Here is an email: {body}\nWhat is a potential subject line for this "
         "email?", "{subject}"),
        ("{body}\nPropose a subject line for this email?", "{subject}"),
        ("This is the content of an email: {body}\nWhat was the subject line "
         "for this email?", "{subject}"),
        ("This is an email\n{body}\n\nWhat is the subject of this email?",
         "{subject}"),
        ("{body}\n\nGenerate a subject line for this email.", "{subject}"),
        ("Write an email with the following subject:\n\n{subject}\n\nEmail:",
         "{body}"),
        ("Write an email with the subject line \"{subject}\".", "{body}"),
    ],
    "cnn_dailymail": [
        ("Write highlights for this article:\n\n{text}\n\nHighlights:",
         "{highlights}"),
        ("Write some highlights for the following "
         "article:\n\n{text}\n\nHighlights:", "{highlights}"),
        ("{text}\n\nWrite highlights for this article.", "{highlights}"),
        ("{text}\n\nWhat are highlight points for this article?",
         "{highlights}"),
        ("{text}\nSummarize the highlights of this article.", "{highlights}"),
        ("{text}\nWhat are the important parts of this article?",
         "{highlights}"),
        ("{text}\nHere is a summary of the highlights for this article:",
         "{highlights}"),
        ("Write an article using the following "
         "points:\n\n{highlights}\n\nArticle:", "{text}"),
        ("Use the following highlights to write an "
         "article:\n\n{highlights}\n\nArticle:", "{text}"),
        ("{highlights}\n\nWrite an article based on these highlights.",
         "{text}"),
    ],
    "gigaword": [
        ("Write a short summary for this text: {text}\n\nSummary:",
         "{summary}"),
        ("Briefly summarize this sentence: {text}\n\nSummary:", "{summary}"),
        ("Generate a short summary this sentence:\n{text}\n\nSummary:",
         "{summary}"),
        ("What is a shorter version of this:\n\n{text}\n\nSummary:",
         "{summary}"),
        ("{text}\n\nWrite a brief summary in a sentence or less.", "{summary}"),
        ("{text}\n\nWhat is a very short summary of the above text?",
         "{summary}"),
        ("{text}\nSummarize the aforementioned text in a single phrase.",
         "{summary}"),
        ("{text}\nCan you generate a short summary of the above paragraph?",
         "{summary}"),
        ("Write a text based on this summary: {summary}\n\nText:", "{text}"),
        ("Write a text based on \"{summary}\"\n\nText:", "{text}"),
    ],
    "multi_news": [
        ("Summarize this article:\n\n{text}\n\nSummary:", "{summary}"),
        ("Write a summary based on this article:\n\n{text}\n\nSummary:",
         "{summary}"),
        ("Article:\n\n{text}\nWhat is a summary?", "{summary}"),
        ("{text}\nWhat is a one-paragraph summary of the above article?",
         "{summary}"),
        ("Here is a news article: {text}\nA summary of this is?", "{summary}"),
        ("News article:\n\n{text}\nWhat is a shorter version of the above "
         "article?", "{summary}"),
        ("{text}\n\nWrite a summary.", "{summary}"),
        ("Article:\n{text}Summary:", "\n{summary}"),
        ("Write an article based on this summary:\n\n{summary}\n\nArticle:",
         "\n{text}"),
        ("{summary}\n\nExpand this summary.", "{text}"),
    ],
    "newsroom": [
        ("{title}\n\n{text}\n\nWrite a one or two sentence summary.",
         "{summary}"),
        ("Please write a short summary for the following "
         "article:\n\n{title}\n\n{text}\n\nSummary:", "{summary}"),
        ("Please briefly summarize this news "
         "article:\n\n{title}\n\n{text}\n\nSummary:", "{summary}"),
        ("{title}\n{text}\nWhat was this article about?", "{summary}"),
        ("{title}\n{text}\nWhat is a short summary of the above article?",
         "{summary}"),
        ("{title}\n\n{text}\nWhat are the most important parts of this text?",
         "{summary}"),
        ("News article: {title}\n\n{text}\nWhat are the most important parts "
         "of this news article?", "{summary}"),
        ("Write an article with the title: \"{title}\"\n\nArticle:",
         "\n{text}"),
        ("Write a title for this article:\n\n{text}\n\nTitle:", "{title}"),
        ("Here is an article:\n\n{text}\n\nWrite a title for it.", "{title}"),
    ],
    "samsum": [
        ("{dialogue}\n\nBriefly summarize that dialogue.", "{summary}"),
        ("Here is a dialogue:\n{dialogue}\n\nWrite a short summary!",
         "{summary}"),
        ("Dialogue:\n{dialogue}\n\nWhat is a summary of this dialogue?",
         "{summary}"),
        ("{dialogue}\n\nWhat was that dialogue about, in two sentences or less?",
         "{summary}"),
        ("Here is a dialogue:\n{dialogue}\n\nWhat were they talking about?",
         "{summary}"),
        ("Dialogue:\n{dialogue}\nWhat were the main points in that "
         "conversation?", "{summary}"),
        ("Dialogue:\n{dialogue}\nWhat was going on in that conversation?",
         "{summary}"),
        ("Write a dialog about anything you want.", "{dialogue}"),
        ("Write a dialog based on this summary:\n{summary}.", "{dialogue}"),
        ("Write a dialog with this premise \"{summary}\".", "{dialogue}"),
    ],
    "xsum": [
        ("Summarize:\n\n{text}\n\nSummary:", "{summary}"),
        ("Summarize this article:\n\n{text}\n\nSummary:", "{summary}"),
        ("Summarize this article in one sentence.\n\n{text}\n\nSummary:",
         "{summary}"),
        ("{text}\nWhat is a summary of this text?", "{summary}"),
        ("{text}\nWhat was that article about?", "{summary}"),
        ("{text}\n\nThis article was about:", "{summary}"),
        ("Article:{text}\n\nA summary of the above article is?", "{summary}"),
        ("Article:{text}\n\nSummarize the main points of that article.",
         "{summary}"),
        ("Write an article based on this summary:\n\n{summary}\n\nArticle:",
         "{text}"),
        ("Write an article based on this \"{summary}\"\n\nArticle:", "{text}"),
    ],
    "squad_v1": [
        ("Please answer a question about the following article about "
         "{title}:\n\n{context}\n\n{question}", "{answer}"),
        ("Read this and answer the question\n\n{context}\n\n{question}",
         "{answer}"),
        ("{context}\n{question}", "{answer}"),
        ("Answer a question about this article:\n{context}\n{question}",
         "{answer}"),
        ("Here is a question about this article: {context}\nWhat is the answer"
         " to this question: {question}", "{answer}"),
        ("Article: {context}\n\nQuestion: {question}", "{answer}"),
        ("Article: {context}\n\nNow answer this question: {question}",
         "{answer}"),
        ("{title}\n{context}\n\nQ: {question}", "{answer}"),
        ("Ask a question about {title}.", "{question}"),
        ("What is the title of this article:\n\n{context}\n\nTitle:",
         "{title}"),
    ],
    "squad_v2": [
        ("{title}:\n\n{context}\n\nPlease answer a question about this "
         "article. If the question is unanswerable, say \"unanswerable\". "
         "{question}", "{answer}"),
        ("Read this and answer the question. If the question is unanswerable, "
         "say \"unanswerable\".\n\n{context}\n\n{question}", "{answer}"),
        ("What is a question about this article? If the question is "
         "unanswerable, say \"unanswerable\".\n\n{context}\n\n{question}",
         "{answer}"),
        ("{context}\n{question} (If the question is unanswerable, say "
         "\"unanswerable\")", "{answer}"),
        ("{context}\nTry to answer this question if possible (otherwise reply "
         "\"unanswerable\"): {question}", "{answer}"),
        ("{context}\nIf it is possible to answer this question, answer it for "
         "me (else, reply \"unanswerable\"): {question}", "{answer}"),
        ("{context}\n\nAnswer this question, if possible (if impossible, reply"
         " \"unanswerable\"): {question}", "{answer}"),
        ("Read this: {context}\n\n{question}\nWhat is the answer? (If it "
         "cannot be answered, return \"unanswerable\")", "{answer}"),
        ("Read this: {context}\nNow answer this question, if there is an "
         "answer (If it cannot be answered, return \"unanswerable\"): "
         "{question}", "{answer}"),
        ("{context}\nIs there an answer to this question (If it cannot be "
         "answered, say \"unanswerable\"): {question}", "{answer}"),
    ],
    "drop": [
        ("Answer based on context:\n\n{context}\n\n{question}", "{answer}"),
        ("{context}\n\nAnswer this question based on the article: {question}",
         "{answer}"),
        ("{context}\n\n{question}", "{answer}"),
        ("{context}\nAnswer this question: {question}", "{answer}"),
        ("Read this article and answer this question {context}\n{question}",
         "{answer}"),
        ("{context}\n\nBased on the above article, answer a question. "
         "{question}", "{answer}"),
        ("Context: {context}\n\nQuestion: {question}\n\nAnswer:", "{answer}"),
        ("Write an article that answers the following question: {question}",
         "{context}"),
        ("Write a question about the following article: {context}\n\nQuestion "
         "about the article:", "{question}"),
        ("{context}\n\nAsk a question about this article.", "{question}"),
    ],
    # Not in FLAN Templates (flan_templates):
    "synth_cot_drop": [
        ("Answer based on context:\n\n{context}\n\n{question}.\n\n"
         "Let's think step by step:", "{cot}\nThe answer is {answer}"),
        ("{context}\n\nAnswer this question by reasoning step-by-step based on "
         "the article: {question}", "{cot} The answer is {answer}"),
        ("{context}\n\n{question}\nStep-by-step reasoning process:",
         "\n{cot} The answer is {answer}"),
        ("{context}\nAnswer this question: {question}. Now, let me think...",
         "\n{cot}\nThe answer is {answer}"),
        ("Read the following article then answer the question. "
         "Explain your answer afterwards.\n{context}\n{question}\n",
         "The answer is {answer}.\nExplanation: {cot}"),
        ("{context}\n\nBased on the above article, answer a question. "
         "{question}\nI need you to give me your thought process first.",
         "{cot}\nThe answer is {answer}"),
        ("Context: {context}\n\nQuestion: {question}\n\nYour thought:",
         "{cot} The answer is {answer}"),
        ("{context}\n{question}\nWhat do you think? I think:",
         "{cot} The answer is {answer}"),
        ("{context} {question} Chain-of-thought:",
         "{cot} The answer is {answer}"),
        ("{context} {question}\n Let's think step by step:\n",
         "{cot} The answer is {answer}"),
    ],
    "quac": [
        ("{background}\n\n{context}\n\nAnswer the following question by taking"
         " a quote from the article: {question}", "{answer}"),
        ("{background}\n\n{context}\n\nUsing a quote from the above article, "
         "answer the following question: {question}", "{answer}"),
        ("Answer by taking a quote from the following "
         "article:\n\n{background}\n\n{context}\n\n{question}", "{answer}"),
        ("{background}\n\n{context}\n\n{question}", "{answer}"),
        ("Background: {background}\nContext: {context}\nQuestion: "
         "{question}\n\nAnswer:", "{answer}"),
        ("Background: {background}\nContext: {context}\nQuestion: {question}. "
         "Whats the answer?", "{answer}"),
        ("{context}\n\nAnswer this question \"{question}\" by extracting the "
         "answer from the text above.", "{answer}"),
        ("{background}\n\n{context} Answer this question using a quote from"
         " the following article:\n\n{question}", "{answer}"),
        ("Which entity is this text "
         "about?\n\n{background}\n\n{context}\n\nEntity:", "{title}"),
        ("{background}\n\n{context}\n\nAsk a question about this article.",
         "{question}"),
    ],
    "para_crawl": [
        ("How do you say \"{sent1}\" in {lang2}?", "{sent2}"),
        ("{sent2} How do you say this sentence in {lang1}?", "{sent1}"),
        ("{sent1} Say this using {lang2}.", "{sent2}"),
        ("Translate from {lang1} to {lang2}:\n\n{sent1}\n\n{lang2}:",
         "{sent2}"),
        ("Translate from {lang2} to {lang1}:\n\n{sent2}\n\n{lang1}:",
         "{sent1}"),
        ("Translate \"{sent2}\" from {lang2} to {lang1}.", "{sent1}"),
        ("Translate \"{sent1}\" to {lang2}.", "{sent2}"),
        ("Translate the following.\n\n{lang1}: {sent1}\n\n{lang2}:", "{sent2}"),
        ("Write a sentence in {lang1}.", "{sent1}"),
        ("Write a sentence in {lang2}.", "{sent2}"),
    ],
    "wmt16_translate": [
        ("{sent1}\n\nTranslate to {lang2}\n\n{lang2}:", "{sent2}"),
        ("{sent2}\n\nTranslate to {lang1}\n\n{lang1}:", "{sent1}"),
        ("{sent2}\n\nCould you please translate this to {lang1}?", "{sent1}"),
        ("{sent2}\n\nTranslate this to {lang1}?", "{sent1}"),
        ("Translate to {lang2}:\n\n{sent1}\n\n{lang2}:", "{sent2}"),
        ("Translate the following sentence to {lang2}:\n{sent1}\n\n{lang2}:",
         "{sent2}"),
        ("How is \"{sent1}\" said in {lang2}?", "{sent2}"),
        ("Translate \"{sent1}\" to {lang2}?", "{sent2}"),
        ("Write a sentence not in {lang1}.", "{sent2}"),
        ("{sent2}\n\nWhich language is this?", "{lang2}"),
    ],
    "wmt14_enfr": [
        ("{sent1}\n\nTranslate to {lang2}.", "{sent2}"),
        ("{sent2}\n\nTranslate to {lang1}.", "{sent1}"),
        ("{sent2}\n\nCould you please translate this to {lang1}?", "{sent1}"),
        ("{sent2}\n\nTranslate this to {lang1}?", "{sent1}"),
        ("Translate to {lang2}:\n\n{sent1}\n\n", "{sent2}"),
        ("Translate the following sentence to {lang2}:\n{sent1}\n\n",
         "{sent2}"),
        ("How is \"{sent1}\" said in {lang2}?", "{sent2}"),
        ("Translate \"{sent1}\" to {lang2}?", "{sent2}"),
        ("Write a sentence not in {lang1}.", "{sent2}"),
        ("{sent2}\n\nWhich language is this?", "{lang2}"),
    ],
    "true_case": [
        ("{lower}\n\nPlease write the text above using proper case.",
         "{answer}"),
        ("{lower}\n\nWrite the above sentence using proper case.", "{answer}"),
        ("{lower}\n\nHow would the previous sentence be correctly capitalized?",
         "{answer}"),
        ("{lower}\nCapitalize this past sentence correctly.", "{answer}"),
        ("{lower}\nRepeat this setence, but with the correct capitalization.",
         "{answer}"),
        ("{lower}\nCan you repeat this sentence, but capitalize it correctly?",
         "{answer}"),
        ("{lower}\n\nThis is all lower case. Can you fix that?", "{answer}"),
        ("{lower}\n\nMake this proper case.", "{answer}"),
        ("Please capitalize where necessary: {lower}", "{answer}"),
        ("{answer}\n\nMake this lower case.", "{lower}"),
    ],
    "fix_punct": [
        ("{no_punct}\n\nAdd punctuation.", "{answer}"),
        ("{no_punct}\n\nCan you repeat this sentence, but add in punctuation?",
         "{answer}"),
        ("{no_punct}\n\nWhat is the correctly punctuated version of this "
         "sentence?", "{answer}"),
        ("{no_punct}\n\nPlease fix the punctuation.", "{answer}"),
        ("{no_punct}\n\nCould you correct the punctuation please?", "{answer}"),
        ("Please add punctuation to this: {no_punct}\n\nPunctuation version:",
         "{answer}"),
        ("Add punctuation: {no_punct}\n\n", "{answer}"),
        ("Add punctuation to the following sentence: {no_punct}\n\n",
         "{answer}"),
        ("Generate a correctly punctuated version of the following text: "
         "{no_punct}\n\n", "{answer}"),
        ("What is the version of the following sentence with correct "
         "punctuation?\n\n{no_punct}\n\n", "{answer}"),
    ],
    "word_segment": [
        ("{no_space}\nGenerate a sentence using the above characters:",
         "{answer}"),
        ("{no_space}\nWhat's a sentence that uses these characters?",
         "{answer}"),
        ("{no_space}\n\nPlease segment the words:", "{answer}"),
        ("Add spaces: {no_space}\n\n", "{answer}"),
        ("Please add spaces between words: {no_space}\n\n", "{answer}"),
        ("This text is missing some spaces, please add them: {no_space}\n\n",
         "{answer}"),
        ("Add spaces between the words in the following text: {no_space}\n\n",
         "{answer}"),
        ("Write the following list of characters into a correctly formed "
         "sentence: {no_space}\n\n", "{answer}"),
        ("{answer}\n\nPlease remove spaces between words.", "{no_space}"),
        ("Remove the spaces from the following sentence: {answer}",
         "{no_space}"),
    ],
    "cosmos_qa": [
        ("{context}\n\nQuestion with options to choose from: "
         "{question}\n{options_}", "{answer}"),
        ("{context}\n\n{options_}\nQ: {question}", "{answer}"),
        ("{context}\n\n{options_}\nAnswer the following question: {question}\n",
         "{answer}"),
        ("{context}\n\nBased on the preceding passage, choose your answer for "
         "question {question}\n{options_}\nThe answer is:", "{answer}"),
        ("{context}\n\nQ with options: Give answer the following question "
         "using evidence from the above passage: {question}\n{options_}",
         "{answer}"),
        ("Context: {context}\nQuestion {question}\nPossible "
         "answers:\n{options_}\nThe answer:", "{answer}"),
        ("Read the following article and answer the question by choosing from "
         "the options.\n\n{context}\n\n{question}\n{options_}...A:",
         "{answer}"),
        ("This question has options. Answer the question about "
         "text:\n\n{context}\n\n{question}\n{options_}", "{answer}"),
        ("Write a question about the following article."
         "\n\n{context}\n\nQuestion:", "{question}\n{options_}"),
        ("{context}\n\nGenerate a question about the above context.",
         "{question}\n{options_}"),
    ],
    # Not in FLAN Templates (flan_templates):
    "synth_cot_cosmos_qa": [
        ("{context}\n\nQuestion: {question}\n{options_}\n"
         "Let's answer step by step.", "{cot} So the answer is {answer}"),
        ("{context}\n\n{options_}\nQ: {question}\nStep by step reasoning:",
         "{cot} The answer is {answer}"),
        ("{context}\n\n{options_}\nLet's answer this carefully: {question}\n",
         "{cot}\nThe answer is {answer}"),
        ("{context}\n\nBased on the preceding passage, answer question "
         "{question}\n{options_}\nLet's solve slowly:",
         "{cot} The answer is {answer}"),
        ("{context}\nSolve the following question "
         "thinking out loud: {question}\n{options_}",
         "{cot} So, the answer is {answer}"),
        ("Context: {context}\nQuestion: {question}\n"
         "\n{options_}\nLet's think:", "{cot}... So the answer is {answer}"),
        ("Read the following article and answer the question."
         "\n{context}\n\n{question}\n{options_}..."
         "Chain-of-thought:", "{cot}\nThe answer is {answer}"),
        ("Answer the question about text:\n\n{context}\n\n{question}\n"
         "{options_}\nCoT:", "{cot} The answer is {answer}"),
        ("{context}\nQuestion: {question}\n{options_}\nChain-of-thought:",
         "{cot} The answer is {answer}"),
        ("Context: {context}\nQ: {question}\n{options_}\nStep-by-step "
         "reasoning process:", "{cot}\nThe answer is {answer}"),
    ],
    "ag_news_subset": [
        ("{title}\n\n{text}\n\nMulti-choice problem: What is this text "
         "about?\n{options_}", "{answer}"),
        ("Choose your answer. {title}\n\n{text}\n\nWhich topic is this article"
         " about?\n{options_}", "{answer}"),
        ("{text}\nQ: Which is the best summary of this article?\n{options_}\nI"
         " think the answer is", "{answer}"),
        ("{text}\nChoose your answer. What is this text "
         "about?\n{options_}\nAnswer:", "{answer}"),
        ("{text}\n\nWhat best summarizes the content of the above "
         "article?\n{options_}", "{answer}"),
        ("Select your answer: Which is this about?\n\n{text}\n\n{options_}",
         "{answer}"),
        ("Select the correct answer: Which is an appropriate title for this "
         "article?\n\n{text}\n\n{options_}", "{answer}"),
        ("Note the options at the end. Select the topic that this "
         "about:\n\n{text}\n\n{options_}", "{answer}"),
        ("Write a title:\n{text}\nTitle:", "{title}"),
        ("{text}\n\nWhat is a good title for this?", "{title}"),
    ],
    "bool_q": [
        ("{text}\n\nSee options at the end. Can we conclude that "
         "{question}?\n\n{options_}", "{answer}"),
        ("{text}\n\nMulti-choice problem: Is it true that "
         "{question}?\n\n{options_}\nThe answer is:", "{answer}"),
        ("{text}\n\n{question}?\n\n{options_}", "{answer}"),
        ("Text: {text}\n\nQuestion: {question}?\n\n{options_}", "{answer}"),
        ("{text}\n\nWhat's the best answer to this question: "
         "{question}?\n\n{options_}...A:", "{answer}"),
        ("{text}\nBased on the above text, what's the best answer to this "
         "question: {question}?\n\n{options_}", "{answer}"),
        ("{text}\nAnswer this question, making sure that the answer is "
         "supported by the text: {question}?\n\n{options_}", "{answer}"),
        ("{text}\n\nChoose your answer: Is the following statement correct "
         "based on the text\n\n{question}\n\n{options_}", "{answer}"),
        ("{title}\n\n{text}\n\n{options_}\nIs this statement correct "
         "\"{question}\"?", "{answer}"),
        ("Is it true that {question} based on the following "
         "text?\n\n{text}\n\n{options_}", "{answer}"),
    ],
    # Not in FLAN Templates (flan_templates):
    "synth_cot_bool_q": [
        ("{passage}\n\nThink out loud. Can we conclude that "
         "{question}?\n\n{options_}", "{cot}. The answer is {answer}"),
        ("{passage}\n\nIs it true that {question}?\n\n{options_}\n"
         "Your thought:", "{cot}. The answer is {answer}"),
        ("{passage}\n\n{question}?\n\n{options_}\nLet's think step by step.",
         "{cot}\nThe answer is {answer}"),
        ("Answer the following question carefully. Think out loud.\n"
         "passage: {passage}\n\nQuestion: {question}?\n\n{options_}",
         "{cot}\nThe answer is {answer}"),
        ("Give the reasoning before answering any question.\n{passage}\n\n"
         "What's the best answer to this question: {question}?\n{options_}...",
         "{cot}. The answer is {answer}"),
        ("{passage}\nBased on the above text, what's the best answer to this "
         "question: {question}?\n{options_}\nLet's think.",
         "{cot}. Final answer: {answer}"),
        ("{passage}\nAnswer this question carefully, making sure that the "
         "answer is supported by the text: {question}?\n\n{options_}"
         "Step-by-step reasoning process:",
         "{cot}. I think the answer is {answer}"),
        ("{passage}\n\nChoose your answer: Is the following statement correct "
         "based on the passage\n\n{question}\n\n{options_}\nChain-of-thought:",
         "{cot}\nThe answer is {answer}"),
        ("{title}\n\n{passage}\n\n{options_}\nIs this statement correct "
         "\"{question}\"? Chain-of-thought:", "{cot}\nThe answer is {answer}"),
        ("Is it true that {question} based on the following "
         "passage?\n\n{passage}\n\n{options_}\nSay why you think so.",
         "{cot}. The answer is {answer}"),
    ],
    "definite_pronoun_resolution": [
        ("{sentence}\n\n{options_}\nWho is {pronoun} referring to?",
         "{answer}"),
        ("{sentence}\n\nWho is \"{pronoun}\" in this prior sentence(see "
         "options)?\n{options_}", "{answer}"),
        ("{sentence}\n\nWho is {pronoun} referring to in this "
         "sentence?\n{options_}", "{answer}"),
        ("Choose your answer: {sentence}\nTell me who {pronoun} is.\n{options_}",
         "{answer}"),
        ("{sentence}\nBased on this sentence, who is {pronoun}?\n\n{options_}",
         "{answer}"),
        ("Choose your answer: Who is {pronoun} in the following "
         "sentence?\n\n{sentence}\n\n{options_}", "{answer}"),
        ("Multi-choice problem: Which entity is {pronoun} this "
         "sentence?\n\n{sentence}\n\n{options_}", "{answer}"),
        ("Who is {pronoun} referring to in the following "
         "sentence?\n{sentence}\n\n{options_}", "{answer}"),
        ("Note that this question lists possible answers. Which person is "
         "{pronoun} referring to in the following "
         "sentence?\n{sentence}\n\n{options_}", "{answer}"),
        ("{sentence}\nWho is \"{pronoun}\"?\n{options_}", "{answer}"),
    ],
    "glue_mrpc": [
        ("Here are two sentences:\n{sentence1}\n{sentence2}\nDo they have the "
         "same meaning?\n{options_}", "{answer}"),
        ("Here are two sentences:\n\n{sentence1}\n\n{sentence2}\nChoose your "
         "answer: are the two sentences saying the same thing?\n{options_}",
         "{answer}"),
        ("{sentence1}\n\n{sentence2}\n\nSelect from the options at the end. Do"
         " the above sentences mean the same thing?\n{options_}", "{answer}"),
        ("{sentence1}\n\n{sentence2}\n\nPlease tell me if the sentences above "
         "mean the same.\n{options_}", "{answer}"),
        ("{sentence1}\n{sentence2}\nSelect from the options at the end. Are "
         "these sentences conveying the same meaning?\n{options_}", "{answer}"),
        ("{sentence1}\n{sentence2}\n(See options at the end). If the first "
         "sentence is true, is the second one also true?\n{options_}",
         "{answer}"),
        ("{sentence1}\n{sentence2}\nAre these two sentences paraphrases of "
         "each other?\n{options_}", "{answer}"),
        ("Do the following two sentences have the same "
         "meaning?\n{sentence1}\n{sentence2}\n\n{options_}\nThe answer is:",
         "{answer}"),
        ("Do these two sentences mean the same "
         "thing?\n{sentence1}\n{sentence2}\n\n{options_}...I think the answer "
         "is", "{answer}"),
        ("Do these sentences have the same "
         "meaning?\n{sentence1}\n{sentence2}\n\n{options_}", "{answer}"),
    ],
    "glue_qqp": [
        ("{question1}\n{question2}\nMulti-choice problem: Would you say that "
         "these questions are the same?\n{options_}", "{answer}"),
        ("{question1}\n{question2}\nDo those questions have the same "
         "meaning?\n{options_}", "{answer}"),
        ("{question1}\n{question2}\n\nMulti-choice problem: Are these two "
         "questions inquiring about the same information?\n{options_}",
         "{answer}"),
        ("{question1}\n\n{question2}\n\nPlease tell me if those questions are "
         "the same.\n{options_}", "{answer}"),
        ("{question1}\n\n{question2}\n\nChoose your answer. Are these two "
         "questions paraphrases of each other?\n{options_}", "{answer}"),
        ("First question: {question1}\nSecond question: {question2}\nAre these"
         " two questions asking the same thing?\n{options_}", "{answer}"),
        ("Question 1: {question1}\nQuestion 2: {question2}\n{options_}\nAre "
         "questions 1 and 2 asking the same thing?", "{answer}"),
        ("Question 1: {question1}\nQuestion 2: {question2}\n{options_}\nWould "
         "the answer to these two questions be the same?", "{answer}"),
        ("Choose from the options at the end. Are the following two questions "
         "the same?\n{question1}\n{question2}\n\n{options_}\nThe answer is:",
         "{answer}"),
        ("Do these questions have the same "
         "meaning?\n{question1}\n{question2}\n\n{options_}", "{answer}"),
    ],
    "imdb_reviews": [
        ("{text}\nChoose your answer. What is the sentiment of this "
         "review?\n{options_}", "{answer}"),
        ("{text}\nWould you say this review is positive or "
         "negative?\n{options_}", "{answer}"),
        ("{text}\nChoose your answer. How would you describe the sentiment of "
         "this review?\n{options_}", "{answer}"),
        ("{text}\n\nIs the sentiment of this review positive or "
         "negative?\n{options_}", "{answer}"),
        ("{text}\n\nDid this review think positively or negatively of the "
         "movie (see options below)?\n{options_}...I think the answer is",
         "{answer}"),
        ("Select the correct sentiment of the following review: "
         "{text}\n{options_}", "{answer}"),
        ("Choose the correct sentiment from "
         "candidates:\n{options_}\n\nTEXT:{text}\n\n", "{answer}"),
        ("Review: {text}\nWhat is the sentiment of this review?\n{options_}",
         "{answer}"),
        ("Review: {text}\nNow, what is this review like?\n{options_}\n",
         "{answer}"),
        ("What's an example of a movie review?", "{text}"),
    ],
    "paws_wiki": [
        ("{sentence1}\n{sentence2}\n\nSelect your answer from the options. Do "
         "these sentences mean the same thing?\n{options_}", "{answer}"),
        ("{sentence1}\n{sentence2}\n\nAre these two sentences paraphrases of "
         "each other?\n{options_}", "{answer}"),
        ("1. {sentence1}\n2. {sentence2}\n\nSelect your answer from the "
         "options. Are these two sentences paraphrases of each "
         "other?\n{options_}...I think the answer is", "{answer}"),
        ("(1) {sentence1}\n(2) {sentence2}\n\nDo these two sentences mean the "
         "same thing?\n\n{options_}", "{answer}"),
        ("Sentence 1: {sentence1}\nSentence 2: {sentence2}\n\nDo these two "
         "sentences convey the same information?\n\n{options_}", "{answer}"),
        ("Do these two sentences from wikipedia have the same "
         "meaning?\n{sentence1}\n{sentence2}\n\n{options_}\nThe answer is:",
         "{answer}"),
        ("Multi-choice question: Same "
         "meaning?\n{sentence1}\n{sentence2}\n\n{options_}", "{answer}"),
        ("Are these paraphrases?\n{sentence1}\n{sentence2}\n\n{options_}",
         "{answer}"),
        ("Do these mean the same?\n{sentence1}\n{sentence2}\n\n{options_}",
         "{answer}"),
        ("Please check if these have the same meaning. {options_}"
         "\n{sentence1}\n{sentence2}", "{answer}"),
    ],
    # Not in FLAN Templates (flan_templates):
    "synth_cot_paws_wiki": [
        ("{sentence1}\n{sentence2}\n\nExplain your answer, do "
         "these sentences mean the same thing?\n{options_}\n"
         "Step-by-step reasoning process:", "{cot} So the answer is {answer}"),
        ("{sentence1}\n{sentence2}\n\nAre these two sentences paraphrases of "
         "each other?\n{options_}\nLet's see.",
         "{cot} So the answer is {answer}"),
        ("1. {sentence1}\n2. {sentence2}\n\n"
         "Are these two sentences paraphrases of each "
         "other?\n{options_}...I think the logic is:",
         "{cot} The answer is {answer}"),
        ("(1) {sentence1}\n(2) {sentence2}\n\nDo these two sentences mean the "
         "same thing?\n\n{options_}\nAhh.", "{cot}. The answer: {answer}"),
        ("Sentence 1: {sentence1}\nSentence 2: {sentence2}\n\nDo these two "
         "sentences convey the same information?\n\n{options_}\nLet's think.",
         "{cot} The answer is {answer}"),
        ("Do these two sentences from wikipedia have the same "
         "meaning?\n{sentence1}\n{sentence2}\n\n{options_}\nThoughts:",
         "{cot}\nAnswer: {answer}"),
        ("Think before you answer: Same "
         "meaning?\n{sentence1}\n{sentence2}\n\n{options_}",
         "{cot}\nThe answer: {answer}"),
        ("Are these paraphrases?\n{sentence1}\n{sentence2}\n\n{options_}\nCoT:",
         "Answer: {answer}"),
        ("Let's carefully answer this question: do these mean the same?\n"
         "{sentence1}\n{sentence2}\n\n{options_}",
         "{cot}\nThe final answer: {answer}"),
        ("Please check if these have the same meaning.\n{options_}\n"
         "{sentence1}\n{sentence2}\nYour thought?",
         "{cot}\nThe answer is {answer}"),
    ],
    "sentiment140": [
        ("{text}\nSelect your answer from the options. What is the sentiment "
         "of this tweet?\n{options_}...I think the answer is", "{answer}"),
        ("{text}\n\nHow would the sentiment of this tweet be "
         "described?\n{options_}", "{answer}"),
        ("{text}\n\nDescribe the sentiment embodied by this "
         "tweet.\n{options_}\nI think the answer is", "{answer}"),
        ("Tweet: {text}\nPredict the sentiment of this tweet.\n{options_}",
         "{answer}"),
        ("Multi-choice question: What is the sentiment of the following "
         "tweet?\nTweet: {text}\n{options_}", "{answer}"),
        ("Select your answer from the options. How would one describe the "
         "sentiment of this tweet?\n{text}\n{options_}", "{answer}"),
        ("Possible tweet sentiments: {options_}\nWrite a tweet that is "
         "{answer}.", "{text}"),
        ("What is an example of a tweet?", "{text}"),
        ("Write a {answer} tweet. Possible tweet types: {options_}", "{text}"),
        ("Sentiment possibilities {options_}. Generate a tweet that has the "
         "following sentiment: {answer} ", "{text}"),
    ],
    # Not in FLAN Templates (flan_templates):
    "synth_cot_sentiment140": [
        ("{text}\nWhat is the sentiment of this tweet?\n{options_}..."
         "I think the solution should be:", "{cot} The answer is {answer}"),
        ("{text}\n\nHow would the sentiment of this tweet be "
         "described?\n{options_}\nStep-by-step reasoning process:",
         "{cot} So the answer is {answer}"),
        ("{text}\n\nDescribe the sentiment embodied by this "
         "tweet.\n{options_}\nThoughts:", "{cot}\nAnswer: {answer}"),
        ("Tweet: {text}\nEXPLAIN the sentiment of this tweet.\n{options_}",
         "Explanation: {cot}\nAnswer: {answer}"),
        ("Think out loud: What is the sentiment of the following "
         "tweet?\nTweet:{text}\n{options_}\n", "{cot} The answer is {answer}"),
        ("Let's think step-by-step to solve this: How would one describe the "
         "sentiment of this tweet?\n{text}\n{options_}\n",
         "Step-by-step reasoning: {cot}\nAnswer: {answer}"),
        ("{text}\nSentiment?\n{options_}\nCoT:", "{cot}\nAnswer: {answer}"),
        ("{text}\nHow is sentiment of the text above?\n{options_}\n"
         "Chain-of-thought:", "{cot}\nAnswer: {answer}"),
        ("{text}\nIs this text positive or negative?\n{options_}\n"
         "Well, I think:", "{cot}\nSo the answer is: {answer}"),
        ("Text: {text}\nIs the text above positive or negative in terms of "
         "sentiment?\n{options_}\nHmm...", "{cot}\nThe answer is: {answer}"),
    ],
    "story_cloze": [
        ("{context}\n{options_}\nWhich option is the next sentence?",
         "{answer}"),
        ("{context}\n\nWhat is the next sentence?\n{options_}", "{answer}"),
        ("{context}\n\nWhat is a natural next sentence?\n{options_}",
         "{answer}"),
        ("{context}\n\nWrite the next sentence, by choosing from:\n{options_}",
         "{answer}"),
        ("Context: {context}\n\nNow do a next sentence "
         "writing task.\n{options_}", "{answer}"),
        ("Story: {context}\n\nIn the options below, what is the most likely to"
         " happen next?\n{options_}", "{answer}"),
        ("Write the next sentence in this story.\n\n{context}\n{options_}",
         "{answer}"),
        ("Choose from options. Continue the following "
         "story.\n\n{context}\n{options_}", "{answer}"),
        ("{options_}\nWrite a story that ends with: {answer}",
         "{context} {answer}"),
        ("Write a plausible story that ends with this sentence?\n\nLast "
         "sentence: {answer}\n{options_}", "{context} {answer}"),
    ],
    "copa": [
        ("{premise} What is the {question}?\n\n{options_}", "{answer}"),
        ("Here is a premise: {premise}\n\nWhat is the {question}?\n\n{options_}",
         "{answer}"),
        ("{premise}\n\nWhat is the {question} of the preceding "
         "sentence?\n\n{options_}", "{answer}"),
        ("{premise}\n\nWhat is a plausible {question}?\n\n{options_}",
         "{answer}"),
        ("Based on the following sentence, what is the "
         "{question}?\n\n{premise}\n\n{options_}", "{answer}"),
        ("{premise}\n\n{question}: \n\n{options_}", "{answer}"),
        ("What is the {question} of the following "
         "sentence?\n\n{premise}\n\n{options_}\nThe answer is:", "{answer}"),
        ("Answer the following question about this "
         "sentence:\n\n{premise}\n\nWhat is the {question}?\n\n{options_}",
         "{answer}"),
        ("Write a sentence.", "{premise}"),
        ("Premise: {premise}\nWhat is the {question}?\n{options_}", "{answer}"),
    ],
    "winogrande": [
        ("How does the sentence end? See options at the "
         "end\n\n{context}\n\n{options_}", "{answer}"),
        ("Write the next sentence.\n\n{context}\n\n{options_}\nAnswer:",
         "{answer}"),
        ("Choose your story that continues the following "
         "story.\n\n{context}\n\n{options_}", "{answer}"),
        ("{options_}\nComplete the following sentence.\n\n{context}\n\n",
         "{answer}"),
        ("Continue writing the following text.\n\n{context}\n\n{options_}",
         "{answer}"),
        ("How does the sentence end?\n\n{context}\n{options_}", "{answer}"),
        ("Write the next sentence.\n\n{context}\n{options_}", "{answer}"),
        ("Continue the following story.\n\n{context}\n{options_}", "{answer}"),
        ("Complete the following sentence.\n\n{context}\n{options_}",
         "{answer}"),
        ("Continue writing the following text.\n\n{context}\n{options_}",
         "{answer}"),
    ],
    # Not in FLAN Templates (flan_templates):
    "synth_cot_winogrande": [
        ("How does the sentence end? Let's give some reasoning before you "
         "answer\n\n{context}\n\n{options_}\n", "{cot} The answer is {answer}"),
        ("Write the next sentence.\n\n{context}\n\n{options_}\n"
         "Chain-of-thought:", "{cot}\nThe answer is {answer}"),
        ("Choose your story that continues the following "
         "story.\n\n{context}\n\n{options_}\nYour thought first:",
         "Thoughts: {cot}\nThe answer is {answer}"),
        ("{options_}\nComplete the following sentence.\n\n{context}\n\nCoT:",
         "{cot}\nThe answer is {answer}"),
        ("Continue writing the following text.\n\n{context}\n\n{options_}\n"
         "Well...", "{cot} So the answer is {answer}"),
        ("How does the sentence end?\n{context}\n{options_}\n"
         "Let's reason step-by-step:", "{cot}... The answer is {answer}"),
        ("Write the next sentence.\n{options_}\n{context}\nStep-by-step "
         "reasoning process:", "{cot}\nThe answer is {answer}"),
        ("Continue the following story. Explain your choice first"
         "\n\n{context}\n{options_}", "{cot}\nThe answer is {answer}"),
        ("Complete the following sentence.\n\n{context}\nLet's think "
         "step-by-step {options_}", "{cot} The answer is {answer}"),
        ("Continue writing the following text. EXPLANATION first!\n{context} "
         "{options_}", "{cot} The answer is {answer}"),
    ],
    "yelp_polarity_reviews": [
        ("{text}\nIs this review positive or negative?\n{options_}\nAnswer:",
         "{answer}"),
        ("{text}\nChoose the sentiment of this review?\n{options_}",
         "{answer}"),
        ("{text}\nChoose: was this review given positively or "
         "negatively?\n{options_}", "{answer}"),
        ("{text}\nHow would this review be described in terms of "
         "sentiment?\n{options_}", "{answer}"),
        ("Choose your answer: is the following review positive or "
         "negative?\n\n{text}\n\n{options_}", "{answer}"),
        ("What is the sentiment of the following review?\n{text}\n{options_}",
         "{answer}"),
        ("How might one describe the sentiment of this review?\n{text}..."
         "{options_} I think the answer is", "{answer}"),
        ("Write a {answer} yelp review ({options_}).", "{text}"),
        ("Possible review types:\n{options_}.\nGenerate a {answer} review "
         "for a place", "{text}"),
        ("{options_} What would be an example of an {answer} review?",
         "{text}"),
    ],
    "arc": [
        ("{question}\n\n{options_}", "{answer}"),
        ("Question: {question}\n{options_}\nAnswer:", "{answer}"),
        ("Question: {question}\n\nWhat is the correct answer to the question "
         "from the following choices?\n{options_}", "{answer}"),
        ("Q: {question}\nWhat is the correct answer to this "
         "question?\n{options_}...A:", "{answer}"),
        ("Choose your answer?\n\n{question}\n\n{options_}", "{answer}"),
        ("Answer the question\n\n{question}\n{options_}", "{answer}"),
        ("{question}\n\nPick the answer from these options\n\n{options_}",
         "{answer}"),
        ("Write a question you would see in a school textbook.", "{question}"),
        ("What's an example of a grad-school level question?", "{question}"),
        ("I just took a test in school today. What question was I asked?",
         "{question}"),
    ],
    "anli": [
        ("{context}\n\nChoose your answer: based on the paragraph above can we"
         " conclude that \"{hypothesis}\"?\n\n{options_}\nI think the answer "
         "is", "{answer}"),
        ("{context}\n\nBased on that paragraph can we conclude that this "
         "sentence is true?\n{hypothesis}\n\n{options_}", "{answer}"),
        ("{context}\n\nCan we draw the following "
         "conclusion?\n{hypothesis}\n\n{options_}", "{answer}"),
        ("{context}\nDoes this next sentence follow, given the preceding "
         "text?\n{hypothesis}\n\n{options_}", "{answer}"),
        ("{context}\nCan we infer the "
         "following?\n{hypothesis}\n\n{options_}\nThe answer is:", "{answer}"),
        ("Read the following paragraph and determine if the hypothesis is "
         "true:\n\n{context}\n\n{options_}\nHypothesis: {hypothesis}\n\n\n",
         "{answer}"),
        ("Read the text and determine if the sentence is true (see options at "
         "the end):\n\n{context}\n\nSentence: {hypothesis}\n{options_}",
         "{answer}"),
        ("Can we draw the following hypothesis from the context (see options)?"
         " \n\nContext:\n\n{context}\n\nHypothesis: {hypothesis}\n{options_}",
         "{answer}"),
        ("Choose from options: Determine if the sentence is true based on the "
         "text below:\n{hypothesis}\n\n{context}\n{options_}", "{answer}"),
        ("Generate a context and a hypothesis.",
         "Context: {context}\n\nHypothesis: {hypothesis}"),
    ],
    # Not in FLAN Templates (flan_templates):
    "synth_cot_anli": [
        ("{premise}\n\nBased on the paragraph above can we"
         " conclude that \"{hypothesis}\"?\n\n{options_}\nI think the chain-of"
         "-thought is", "{cot}. The answer is {answer}"),
        ("{premise}\n\nBased on that paragraph can we conclude that this "
         "sentence is true?\n{hypothesis}\n\n{options_}\nLet's think step by "
         "step:", "{cot} The answer is {answer}"),
        ("{premise}\n\nCan we draw the following "
         "conclusion?\n{hypothesis}\n\n{options_}\nHmmm, let's see.",
         "{cot} The answer is {answer}"),
        ("{premise}\nDoes this next sentence follow, given the preceding "
         "text?\n{hypothesis}\n\n{options_}\nLet me think first.",
         "{cot} The answer is {answer}"),
        ("{premise}\nCan we infer the "
         "following?\n{hypothesis}\n\n{options_}\nI think:",
         "{cot} The answer is {answer}"),
        ("Read the following paragraph and determine if the hypothesis is "
         "true:\n\n{premise}\n\n{options_}\nHypothesis: {hypothesis}\n\nLet's "
         "think before answering.", "{cot} The answer is {answer}"),
        ("Read the text and determine if the sentence is true (let's think "
         "step by step first):\n\n{premise}\n\nSentence: "
         "{hypothesis}\n{options_}", "{cot} The answer is {answer}"),
        ("Think carefully before answering: can we draw the following "
         "hypothesis from the premise\nContext:\n\n{premise}\n\nHypothesis: "
         "{hypothesis}\n{options_}",
         "Let me think. {cot} The answer is {answer}"),
        ("Determine if the sentence is true based on the text below:\n"
         "{hypothesis}\n\n{premise}\n{options_}\n"
         "Step-by-step reasoning process:", "{cot} The answer is {answer}"),
        ("Generate a premise and a hypothesis, together with explanation",
         "Context: {premise}\nHypothesis: {hypothesis}\n{options_}\n"
         "Explanation: {cot} The answer is {answer}"),
    ],
    "coqa": [
        ("{text}\n\nAnswer the following "
         "questions:\n{numbered_questions}\n\nNumbered answers:",
         "{numbered_answers}"),
        ("Read the text and answer the "
         "questions.\n\n{text}\n\n{numbered_questions}\n\nNumbered answers:",
         "{numbered_answers}"),
        ("Answer the questions at the end based on the "
         "text.\n\n{text}\n\n{numbered_questions}\n\nNumbered answers:",
         "{numbered_answers}"),
        ("{text}\n\nAnswer this series of "
         "questions:\n\n{numbered_questions}\n\nNumbered answers:",
         "{numbered_answers}"),
        ("{text}\n\nWhat are the answers to this following set of "
         "questions:\n\n{numbered_questions}\n\nNumbered answers:",
         "{numbered_answers}"),
        ("{text}\n\nNow, provide a numbered list of answers to these "
         "questions:\n\n{numbered_questions}\n\nNumbered answers:",
         "{numbered_answers}"),
        ("{text}\n\n{numbered_questions}\n\nNumbered answers:",
         "{numbered_answers}"),
        ("{text}\n\n{numbered_questions}\n\nProvide a numbered list of "
         "answers.", "{numbered_answers}"),
        ("Make use of the article to answer the "
         "questions.\n\n{text}\n\n{numbered_questions}\n\nNumbered answers:",
         "{numbered_answers}"),
        ("{text}\n\nBased on the article and the following list of answers, "
         "write a list of questions.\n\n{numbered_answers}\n\nNumbered "
         "questions:", "{numbered_questions}"),
    ],
    "opinion_abstracts_rotten_tomatoes": [
        ("{numbered_reviews}\n\nWrite a one sentence summary of the reviews "
         "above.", "{critic_consensus}"),
        ("{numbered_reviews}\n\nWhat is a brief summary of "
         "the following reviews?", "{critic_consensus}"),
        ("{numbered_reviews}\nBased on these individual reviews, what is the "
         "critic consensus?", "{critic_consensus}"),
        ("{numbered_reviews}\nWhat is the consensus?", "{critic_consensus}"),
        ("Here are some reviews for a movie: {numbered_reviews}\n\nWhat was "
         "the overall consensus about the movie?", "{critic_consensus}"),
        ("Summarize the following movie "
         "reviews:\n\n{numbered_reviews}\n\nSummary:", "{critic_consensus}"),
        ("Write a one sentence review of the movie \"{movie}\".",
         "{critic_consensus}"),
        ("Write an ordered list of reviews about \"{movie}\".",
         "{numbered_reviews}"),
        ("The critic consesnsus is: {critic_consensus}. What reviews supported"
         " this critic consensus?", "{numbered_reviews}"),
        ("Which movie is the following review "
         "about?\n\n{first_review}\n\nMovie:", "{movie}"),
    ],
    "opinion_abstracts_idebate": [
        ("{argument_sentences}\n\nWhat is the general argument implied by "
         "these sentences?", "{claim}"),
        ("Sentences: {argument_sentences}\n\nWhat claim can be made from these"
         " sentences?", "{claim}"),
        ("{debate_name}\nWhat argument could one make about this debate topic?",
         "{claim}"),
        ("{debate_name}\nWhat is a possible side to this debate?", "{claim}"),
        ("What claim can be made from the following pieces of "
         "evidence?\n\n{argument_sentences}", "{claim}"),
        ("Summarize the argument implied by these "
         "sentences?\n\n{argument_sentences}", "{claim}"),
        ("What debate topic are the following sentences "
         "about?\n\n{argument_sentences}", "{debate_name}"),
        ("What is the debate topic for the following "
         "sentences?\n\n{argument_sentences}", "{debate_name}"),
        ("{claim}\nCome up with some evidence to support this claim.",
         "{argument_sentences}"),
        ("Claim: {claim}\nWhat evidence supports this claim?",
         "{argument_sentences}"),
    ],
    "common_gen": [
        ("Concepts: {concepts}\n\nWrite a sentence that includes all these "
         "words.", "{target}"),
        ("Keywords: {concepts}\n\nWhat is a sentence that includes all these "
         "keywords?", "{target}"),
        ("Here are some concepts: {concepts}\n\nWhat is a sentence about these"
         " concepts?", "{target}"),
        ("Produce a sentence which mentions all of these concepts: {concepts}",
         "{target}"),
        ("Write a sentence about the following things:\n\n{concepts}",
         "{target}"),
        ("Generate a sentence that includes all the following words: {concepts}",
         "{target}"),
        ("What are the keywords in the following sentence:\n\n{target}",
         "{concepts}"),
        ("What are the most important words in the following "
         "sentence:\n\n{target}", "{concepts}"),
        ("Identify the most salient words in this sentence:\n\n{target}",
         "{concepts_newline}"),
        ("Generate a sentence, and then tell me the concepts included in that "
         "sentence.", "Sentence:\n{target}\n\nConcepts:\n{concepts_newline}"),
    ],
    # Not in FLAN Templates (flan_templates):
    "synth_cot_common_gen": [
        ("Concepts: {concepts}\n\nWrite a sentence that includes all these "
         "words. Chain-of-thought:", "{cot} The answer is {target}"),
        ("Keywords: {concepts}\n\nWhat is a sentence that includes all these "
         "keywords? Let see...", "{cot} The answer is {target}"),
        ("Here are some concepts: {concepts}\n\nWhat is a sentence about these"
         " concepts? Hm...", "{cot}\nThe answer is {target}"),
        ("Produce a sentence which mentions all of these concepts: {concepts} "
         "Let's reason first:", "{cot}\nThe answer is {target}"),
        ("Write a sentence about the following things:\n\n{concepts}\n"
         "Thoughts:", "{cot}\nThe answer is {target}"),
        ("Generate a sentence that includes all the following words (thinking "
         "out loud): {concepts}", "{cot}\nThe answer is {target}"),
        ("Let's give an explanable answer to this question: generate a "
         "sentence using words: {concepts}", "{cot} The answer is {target}"),
        ("Think step-by-step to answer this question: generate a "
         "sentence using concepts: {concepts}\n"
         "Step-by-step reasoning process:", "{cot} The answer is {target}"),
        ("Think step-by-step to answer this question: generate a "
         "sentence using concepts: {concepts_newline}",
         "\n{cot} The answer is {target}"),
        ("Answer this question: generate a sentence using concepts: "
         "{concepts_newline}. Think step-by-step:",
         "\n{cot}\nThe answer is {target}"),
    ],
    "dart": [
        ("Triple: {tripleset}\n\nWhat is a sentence that describes this triple?",
         "{target}"),
        ("Data: {tripleset}\n\nWhat would a sentence about this data be like?",
         "{target}"),
        ("Generate an approximately fifteen-word sentence that describes all "
         "this data: {tripleset}\n\n", "{target}"),
        ("Here is some data: {tripleset}.\n\nWrite a sentence that describes "
         "this data:", "{target}"),
        ("This is some data: {tripleset}.\n\nGenerate a detailed description "
         "of this data.", "{target}"),
        ("Generate a sentence about this data: {tripleset}\nSentence:",
         "{target}"),
        ("Write a sentence that about [{tripleset}].", "{target}"),
        ("Produce a long descriptive sentence that uses all these words: "
         "{tripleset}", "{target}"),
        ("What concepts are described in the following "
         "sentence?\n\n\"{target}\"\n\nReturn the answer as pairs of triples.",
         "{tripleset_newline}"),
        ("Create a set of triples that describes the content in the following "
         "sentence.\n\n{target}\n\n", "{tripleset_newline}"),
    ],
    "e2e_nlg": [
        ("Attributes: {meaning_representation}. Produce a detailed sentence "
         "about this restaurant.", "{target}"),
        ("Data: {meaning_representation}. Can you generate a sentence about "
         "this data?", "{target}"),
        ("Data: {meaning_representation}. What is a sentence that describe "
         "this data?", "{target}"),
        ("Here are some keywords about a "
         "restaurant:\n\n{meaning_representation}. Write a sentence that "
         "describes the following attributes of a restaurant.", "{target}"),
        ("Here is some data about a restaurant: {meaning_representation}. "
         "Write a sentence that includes the above data about a restaurant",
         "{target}"),
        ("Sentence: {meaning_representation}\n\nCan you represent the content "
         "in this sentence in data form?", "{target}"),
        ("Write a sentence about a restaurant with all the following "
         "attributes: {meaning_representation}\nSentence:", "{target}"),
        ("Write a sentence that is about a restaurant with all the following "
         "properties: {meaning_representation}\nSentence:", "{target}"),
        ("Produce a detailed sentence about a restaurant using the following "
         "words: {meaning_representation}\nSentence:", "{target}"),
        ("Generate a descriptive sentence about a restaurant using the "
         "following words:\n\n{meaning_representation}\nSentence:", "{target}"),
    ],
    "web_nlg_en": [
        ("{input_string}\n\nWhat is sentence that verbalizes this data?",
         "{target}"),
        ("Data: {input_string}\n\nSentence about the following data: ",
         "{target}"),
        ("Here is some data: {input_string}.\n\nWrite a sentence that "
         "describes this data.\nSentence:", "{target}"),
        ("This is some data: {input_string}.\n\nGenerate a detailed "
         "description of this data.\nSentence:", "{target}"),
        ("Generate a sentence about this data: {input_string}.\nSentence:",
         "{target}"),
        ("Generate a sentence that describes the following data: "
         "{input_string}.\nSentence:", "{target}"),
        ("Produce a long descriptive sentence that uses all these words: "
         "{input_string}.\nSentence:", "{target}"),
        ("Generate an approximately fifteen-word sentence that describes all "
         "this data: {input_string}.\nSentence:", "{target}"),
        ("Sentence: {target}\n\nWhat data can be extracted from this sentence?",
         "{input_string}"),
        ("Sentence: {target}\n\nWhat structured data could we extract from "
         "this sentence?", "{input_string}"),
    ],
    "wiki_lingua_english_en": [
        ("{source}\n\nSummary:", "{target}"),
        ("Summarize the following:\n{source}\n\nSummary:", "{target}"),
        ("Summarize this article:\n\n{source}\n\nSummary:", "{target}"),
        ("Summarize this article in one sentence.\n{source}\n\nSummary:",
         "{target}"),
        ("What is a one-sentence summary of the following "
         "article?\n{source}\n\nSummary:", "{target}"),
        ("In one sentence, describe what the following article is "
         "about:\n\n{source}\n\nSummary:", "{target}"),
        ("Article: {source}\n\nWhat is a summary?", "{target}"),
        ("Article: {source}\nWhat is a summary of what this article is about?",
         "{target}"),
        ("Write an article based on this summary:\n\n{target}\n\nArticle:",
         "{source}"),
        ("Write an article based on this \"{target}\"\n\nArticle:", "{source}"),
    ],
    "multirc": [
        ("{paragraph}\n\nQuestion: \"{question}\"\n\nResponse: "
         "\"{response}\"\n{options_}\nDoes the response correctly answer the "
         "question?\n\n", "{answer}"),
        ("{paragraph}\n\nQuestion: \"{question}\"\n\nResponse: "
         "\"{response}\"\n\nBased on the paragraph, is the response to the "
         "question is factually correct?\n\n{options_}", "{answer}"),
        ("{paragraph}\n\nQuestion: \"{question}\"\n\nAnswer: "
         "\"{response}\"\n\nIs this answer correct?\n\n{options_}...I think "
         "the answer is", "{answer}"),
        ("Paragraph: {paragraph}\n\nQuestion: \"{question}\"\n\nAnswer: "
         "\"{response}\"\n\nBased on the paragraph, choose if the answer is "
         "correct:\n\n{options_}", "{answer}"),
        ("{paragraph}\n\nChoose from options: Based on the paragraph, does the"
         " response \"{response}\" correctly answer the question "
         "\"{question}\"?\n\n{options_}", "{answer}"),
        ("{paragraph}\n\nChoose your answer: According to the above paragraph,"
         " the correct answer to the question \"{question}\" is "
         "\"{response}\"?\n\n{options_}", "{answer}"),
        ("{paragraph}\n\nAfter reading the above, is \"{response}\" the "
         "correct answer to the question \"{question}\"?\n\n{options_}",
         "{answer}"),
        ("{paragraph}\n\nQuestion: \"{question}\"\n\nAnswer: "
         "\"{response}\"\n\nIs this answer to the question correct?"
         "\n{options_}", "{answer}"),
        ("{paragraph}\nDo you have any questions?", "{question}"),
        ("{paragraph}\nWhat question would one ask from this paragraph?",
         "{question}"),
    ],
    "cb": [
        ("{premise}\n\nBased on the paragraph above can we conclude that "
         "\"{hypothesis}\"?\n\n{options_}", "{answer}"),
        ("{premise}\n\nBased on that paragraph can we conclude that this "
         "sentence is true?\n{hypothesis}\n\n{options_}", "{answer}"),
        ("{premise}\n\nCan we draw the following conclusion (choose your "
         "answer)?\n{hypothesis}\n\n{options_}", "{answer}"),
        ("{premise}\nSelect from options. Does this next sentence follow, "
         "given the preceding text?\n{hypothesis}\n\n{options_}", "{answer}"),
        ("{premise}\nMulti-choice question: Can we infer the "
         "following?\n{hypothesis}\n\n{options_}", "{answer}"),
        ("Multi-choice problem: Read the following paragraph and determine if "
         "the hypothesis is true:\n\n{premise}\n\nHypothesis: "
         "{hypothesis}\n{options_}", "{answer}"),
        ("You will be given options, read the text and determine if the "
         "sentence is true:\n\n{premise}\n\nSentence: "
         "{hypothesis}\n{options_}", "{answer}"),
        ("Can we draw the following hypothesis from the context? "
         "\n\nContext:\n\n{premise}\n\nHypothesis: {hypothesis}\n{options_}",
         "{answer}"),
        ("Determine if the sentence is true based on the text "
         "below:\n{hypothesis}\n{options_}\n{premise}\n", "{answer}"),
        ("Generate a context and a hypothesis.",
         "Context: {premise}\n\nHypothesis: {hypothesis}"),
    ],
    "cola": [
        ("Sentence: \"{sentence}\"\nPick from options: would a linguist rate "
         "this sentence to be acceptable linguistically?\n\n{options_}...I "
         "think the answer is", "{answer}"),
        ("{sentence}\n\nHow would you consider the linguistic integrity of the"
         " preceding sentence?\n{options_}", "{answer}"),
        ("Test sentence: \"{sentence}\"\nIs this test sentence a correct "
         "grammatical English sentence?\n\n{options_}", "{answer}"),
        ("Sentence: \"{sentence}\"\nWould a linguist rate this sentence to be "
         "acceptable linguistically?\n\n{options_}", "{answer}"),
        ("Choose from options, is the following sentence linguistically "
         "acceptable?\n{sentence}\n{options_}", "{answer}"),
        ("Choose from the possible answers, would the following sentence, by "
         "the strictest standards, be considered correct by a "
         "linguist?\n\n{sentence}\n{options_}", "{answer}"),
        ("Multi-choice problem: Is the next sentence syntactically and "
         "semantically acceptable?\n\n{sentence}\n{options_}", "{answer}"),
        ("Would a linguist find the following sentence to be a valid English "
         "sentence grammatically?\n\n{sentence}\n{options_}", "{answer}"),
        ("Generate short a sentence that can be linguistically classified as "
         "{answer} ({options_})", "{sentence}"),
        ("Produce a brief English sentence that would be considered "
         "grammatically as category: {answer}\nAll categories: {options_}",
         "{sentence}"),
    ],
    "sst2": [
        ("Review:\n{sentence}\nIs this movie review sentence negative or "
         "positive?\n{options_}\nThe answer is:", "{answer}"),
        ("{options_}\nShort movie review: {sentence}\nDid the critic thinking "
         "positively or negatively of the movie?\n\n", "{answer}"),
        ("Sentence from a movie review: {sentence}\nSelect your answer: was "
         "the movie seen positively or negatively based on the preceding "
         "review?\n\n{options_}", "{answer}"),
        ("\"{sentence}\"\nHow would the sentiment of this sentence be "
         "perceived --\n\n{options_}\nAnswer:", "{answer}"),
        ("Is the sentiment of the following sentence positive or negative (see"
         " options at the end)?\n{sentence}\n{options_}", "{answer}"),
        ("What is the sentiment of the following movie (choose your answer "
         "from the options) review sentence?\n{sentence}\n{options_}\nThe "
         "answer is:", "{answer}"),
        ("{options_}Would the following phrase be considered positive or "
         "negative?\n\n{sentence}\n", "{answer}"),
        ("Does the following review have a positive or negative opinion of the"
         " movie?\n\n{sentence}\n{options_}", "{answer}"),
        ("Write a \"{answer}\" movie review ({options_}).", "{sentence}"),
        ("Generate a short movie review that has \"{answer}\" sentiment "
         "({options_}).", "{sentence}"),
    ],
    "mnli": [
        ("Premise: {premise}\n\nHypothesis: {hypothesis}\n\nDoes the premise "
         "entail the hypothesis?\n\n{options_}", "{answer}"),
        ("Premise: {premise}\nHypothesis: {hypothesis}\nIs the hypothesis "
         "entailed by the premise?\n{options_} And the answer is:", "{answer}"),
        ("Here is a premise:\n{premise}\n\nHere is a "
         "hypothesis:\n{hypothesis}\n\nHere are the options: {options_}\nIs it"
         " possible to conclude that if the premise is true, then so is the "
         "hypothesis?\n", "{answer}"),
        ("Sentence 1: {premise}\n\nSentence 2: {hypothesis}\n{options_}\nIs "
         "this second sentence entailed by the first sentence?\n\n",
         "{answer}"),
        ("See the multi-choice question below:\n\nSentence 1: "
         "{premise}\n\nSentence 2: {hypothesis}\n\nIf the first sentence is "
         "true, then is the second sentence true?\n{options_}", "{answer}"),
        ("Based on the premise \"{premise}\", can we conclude the hypothesis "
         "\"{hypothesis}\" is true (see options)?\n\n{options_}", "{answer}"),
        ("Choose your answer from options. Premise: \"{premise}\" If this "
         "premise is true, what does that tell us about whether it entails the"
         " hypothesis \"{hypothesis}\"?\n\n{options_}", "{answer}"),
        ("Premise:\n\"{premise}\" Based on this premise, is the hypothesis "
         "\"{hypothesis}\" true?\n{options_}", "{answer}"),
        ("If {premise}, can we conclude that \"{hypothesis}\"?\n{options_}",
         "{answer}"),
        ("{premise}\n\nDoes it follow that \"{hypothesis}\"?\n{options_}",
         "{answer}"),
    ],
    "qnli": [
        ("Does the sentence \"{sentence}\" answer the question "
         "\"{question}\"\n\n{options_}", "{answer}"),
        ("Single/multi-select question: Does the sentence \"{sentence}\" "
         "provide a valid answer to the question \"{question}\"\n{options_}",
         "{answer}"),
        ("Choose your answer: Is \"{sentence}\" a good answer to the question "
         "\"{question}\"\n{options_}", "{answer}"),
        ("{options_}\nDoes \"{sentence}\" correctly answer the question of "
         "{question}\n\n", "{answer}"),
        ("Choose your reply from the options at the end. Does \"{sentence}\" "
         "contain the correct answer to \"{question}\"\n{options_}",
         "{answer}"),
        ("Q: {question}\n A: {sentence}\n Does the answer correctly answer the"
         " question\n\n{options_}", "{answer}"),
        ("Question: {question}\nAnswer: {sentence}\n A single-select problem: "
         "Is the question answered in a satisfactory fashion?\n\n{options_}",
         "{answer}"),
        ("Question: {question}\n\nIs {sentence} a good answer to this "
         "question?\n\n{options_}", "{answer}"),
        ("Question: {question}\n\nIs \"{sentence}\" the correct answer?\n"
         "{options_}", "{answer}"),
        ("Can you generate a question with a factual answer?", "{question}"),
    ],
    "wnli": [
        ("If \"{sentence1}\", can we conclude that "
         "\"{sentence2}\"\n{options_}\nI think the answer is", "{answer}"),
        ("If \"{sentence1}\", does it follow that \"{sentence2}\"\n{options_}",
         "{answer}"),
        ("If \"{sentence1}\", is \"{sentence2}\" "
         "correct?\n\n{options_}\nAnswer:", "{answer}"),
        ("Multi-select: Let's say that \"{sentence1}\"\n\nCan we now say that "
         "\"{sentence2}\"?\n\n{options_}", "{answer}"),
        ("\"{sentence1}\" is a true sentence.\n\nDoes this mean that "
         "\"{sentence2}\"?\n\n{options_}", "{answer}"),
        ("Does \"{sentence2}\" appear to be an accurate statement based on "
         "\"{sentence1}\"?\n\n{options_}", "{answer}"),
        ("Can we conclude that \"{sentence2}\" if the statement "
         "\"{sentence1}\" is true?\n\n{options_}", "{answer}"),
        ("Multi-select: Is it possible to draw the conclusion that "
         "\"{sentence2}\" if \"{sentence1}\"?\n\n{options_}", "{answer}"),
        ("Multi-choice problem: Is \"{sentence2}\" true if "
         "\"{sentence1}\"?\n\n{options_}", "{answer}"),
        ("Sentence 1: \"{sentence1}\"\n\n Sentence 2: \"{sentence2}\"\n\nIs "
         "sentence 2 true, based on sentence 1?\n{options_}", "{answer}"),
    ],
    "snli": [
        ("If \"{premise}\", does this mean that \"{hypothesis}\"?\n\n{options_}",
         "{answer}"),
        ("Single/multi-select question: If \"{premise}\", can we conclude "
         "\"{hypothesis}\"?\n\n{options_}", "{answer}"),
        ("Choose your answer: If \"{premise}\", does it logically follow that "
         "\"{hypothesis}\"?\n\n{options_}", "{answer}"),
        ("Multi-choice problem: Based on the sentence \"{premise}\", is the "
         "sentence \"{hypothesis}\" a true sentence?\n\n{options_}",
         "{answer}"),
        ("Premise: {premise}\n\nHypothesis: {hypothesis}\n\n.Multi-select "
         "problem: Can we conclude that the hypothesis is true if the premise "
         "is true?\n\n{options_}", "{answer}"),
        ("Premise: {premise}\n\nHypothesis: {hypothesis}\n\n.Choose the "
         "correct answer: Given the premise, can we conclude the "
         "hypothesis?\n\n{options_}", "{answer}"),
        ("Here is a premise: \"{premise}\"\n\nHere is a hypothesis: "
         "\"{hypothesis}\"\n\n.Does the premise tell us whether the hypothesis"
         " is true?\n\n{options_}", "{answer}"),
        ("Single/multi-select question: Is it possible to conclude that "
         "\"{premise}\" if \"{hypothesis}\"?\n\n{options_}...I think the "
         "answer is", "{answer}"),
        ("Is the premise \"{premise}\" true if \"{hypothesis}\"?\n{options_}",
         "{answer}"),
        ("Write a brief sentence.", "{hypothesis}"),
    ],
    "trec": [
        ("What type of thing is the question \"{text}\" asking "
         "about?\n\n{options_}\nAnswer:", "{answer}"),
        ("Is the question \"{text}\" asking about an entity, an abbreviation, "
         "a description, a human, a location, or a numeric "
         "entity?\n\n{options_}", "{answer}"),
        ("{options_}Would the answer to the question \"{text}\" be an entity, "
         "an abbreviation, a description, a human, a location, or a numeric "
         "value?\n\n", "{answer}"),
        ("This is a question with answer options. What kind of thing would the"
         " answer to the question \"{text}\" be an entity, an abbreviation, a "
         "description, a human, a location, or a numeric value?\n\n{options_}",
         "{answer}"),
        ("Choose your answer: What is \"{text}\" asking "
         "about?\n\n{options_}\nAnswer:", "{answer}"),
        ("From the following options, what is the question \"{text}\" asking "
         "about?\n\n{options_}", "{answer}"),
        ("{text}\n\nWhat kind of thing would answer this "
         "question?\n\n{options_}", "{answer}"),
        ("Here is a single or multi-choice question: {text}\n\nWould the "
         "answer to this question be an entity, an abbreviation, a "
         "description, a human, a location, or a numeric value?\n\n{options_}",
         "{answer}"),
        ("Q: {text}\n\nWhich one of the following options would the answer to "
         "this be?\n\n{options_}\n\nA:", "{answer}"),
        ("Please ask me a question.", "{text}"),
    ],
    "stsb": [
        ("{sentence1}\n{sentence2}\n\nRate the textual similarity of these two"
         " sentences on a scale from 0 to 5, where 0 is \"no meaning overlap\""
         " and 5 is \"means the same thing\".\n\n{options_}", "{answer}"),
        ("{sentence1}\n{sentence2}\n\nOn a scale from 0 to 5, where 0 is \"no "
         "meaning overlap\" and 5 is \"means the same thing\", how closely "
         "does the first sentence resemble the second one?\n\n{options_}",
         "{answer}"),
        ("Sentence 1: {sentence1}\n\n Sentence 2: {sentence2}\n\nFrom 0 to 5 "
         "(0=\"no meaning overlap\" and 5=\"means the same thing\"), how "
         "similar are the two sentences?\n\n{options_}", "{answer}"),
        ("Select from options: How similar are the following two "
         "sentences?\n\n{sentence1}\n{sentence2}\n\nGive the answer on a scale"
         " from 0 - 5, where 0 is \"not similar at all\" and 5 is \"means the "
         "same thing\".\n\n{options_}", "{answer}"),
        ("Single/multi-select question: Do the following sentences say the "
         "same thing?\n\n{sentence1}\n{sentence2}\n\nReturn your answer on a "
         "scale from 0 to 5, where 0 is \"not similar\" and 5 is \"very "
         "similar\".\n\n{options_}", "{answer}"),
        ("Rate the similarity of the following two sentences on a scale from 0"
         " to 5, where 0 is \"no meaning overlap\" and 5 is \"means the same "
         "thing\"?\n\n{sentence1}\n{sentence2}\n\n{options_}", "{answer}"),
        ("On a scale from 0-5, where 0 is \"not similar\" and 5 is \"very "
         "similar\", how similar is the sentence \"{sentence1}\" to the "
         "sentence \"{sentence2}\"?\n\n{options_}", "{answer}"),
        ("How similar are these two sentences, on a scale from 0-5 (0 is \"not"
         " similar\" and 5 is \"very "
         "similar\")?\n\n{sentence1}\n{sentence2}\n\n{options_}", "{answer}"),
        ("{sentence1}\n\nGenerate a new sentence that is, on a scale from 0 to"
         " 5, a {answer} in textual similarity to the above sentence.",
         "{sentence2}"),
        ("{sentence2}\n\nWhat is a sentence that would be (on a scale from 0 "
         "to 5) a {answer} out of 5 in terms of textual similarity to the "
         "above sentence?", "{sentence1}"),
    ],
    "hellaswag": [
        ("What happens next in this paragraph?\n\n{context}\n{options_}",
         "{answer}"),
        ("Multi-choice problem: Continue writing the next sentence in this "
         "paragraph:\n\n{context}\n\n{options_}", "{answer}"),
        ("Select from options: Continue writing the next "
         "sentence.\n\n{context}\n\n{options_}\nAnswer:", "{answer}"),
        ("This is a test of commonsense with single/multi-choices. Complete "
         "the next sentence:\n\n{context}\n\n{options_}\nThe answer is:",
         "{answer}"),
        ("Write the next sentence in this paragraph:\n\n{context}\n\n{options_}",
         "{answer}"),
        ("Multi-select problem: How does the next paragraph "
         "end?\n\n{context}\n\n{options_}", "{answer}"),
        ("{options_}Choose from options above and answer: What most naturally "
         "follows?\n\n{context}\nAnswer:", "{answer}"),
        ("What happens next?\n\n{context}\n\n{options_}", "{answer}"),
        ("What is the most logical next event?\n\n{context}\n\n{options_}",
         "{answer}"),
        ("Write the next sentence in the following "
         "story.\n\n{context}\n\n{options_}. The answer should be", "{answer}"),
    ],
    "piqa": [
        ("Here is a goal: {goal}\n\nHow would you accomplish this "
         "goal?\n\n{options_}", "{answer}"),
        ("Here is a goal: {goal}\n\nWhich way makes more sense to accomplish "
         "this goal?\n\n{options_}", "{answer}"),
        ("This is a question with answer options. Goal: {goal}\n\nWhich of the"
         " following methods is more reasonable for accomplishing this "
         "goal?\n\n{options_}...I think the answer is", "{answer}"),
        ("Objective: {goal}\n\nWhich of the following solutions is more sound "
         "in terms of naive physics reasoning?\n\n{options_}", "{answer}"),
        ("Multi-choice problem: Choose from the options at the end, and answer"
         " how do you do this: {goal}\n\n{options_}", "{answer}"),
        ("What is the best way to: {goal}\n\n{options_}\nAnswer:", "{answer}"),
        ("Single/multi-choice problem: Which of the following solutions is "
         "better for the following goal:\n{goal}\n\n{options_}", "{answer}"),
        ("This question has options. How would someone go about accomplishing "
         "this goal?\n{goal}\n\n{options_}", "{answer}"),
        ("What's an example of a task that requires knowledge of physical "
         "objects to perform?", "{goal}"),
        ("What kind of task would test someone's ability to perform physical "
         "reasoning?", "{goal}"),
    ],
    "openbookqa": [
        ("{fact}\n{question}\n\n{options_}", "{answer}"),
        ("This question has options. Select from options: Read this fact: "
         "\"{fact}\"\n\nNow answer this question: \"{question}\"\n\n{options_}",
         "{answer}"),
        ("Given the fact \"{fact}\", what is the answer to the question or "
         "completion \"{question}\"\n\n{options_}", "{answer}"),
        ("Multi-select: Knowing that \"{fact}\", how would one answer "
         "\"{question}\"\n\n{options_}...A:", "{answer}"),
        ("Use evidence from the fact that {fact} to answer the following "
         "question. Choose from options. \"{question}\"\n\n{options_}",
         "{answer}"),
        ("Fact: {fact}\nQuestion: {question}\n\nWhat's the answer? {options_}",
         "{answer}"),
        ("Use this fact to answer the question: {fact}\n\n{question}\n\n"
         "{options_}\n\nThe answer is:", "{answer}"),
        ("What sentence would provide a factual answer to this question: "
         "\"{question}\"", "{fact}"),
        ("What is a random fact?", "{fact}"),
        ("Generate a sentence that contains a fact.", "{fact}"),
    ],
    # Not in FLAN Templates (flan_templates):
    "lambada": [
        ("{sentence}", "{answer}"),
        ("Complete the following text: {sentence}", "{answer}"),
        ("\"{sentence} _ ...\" What is the word in the blank space (_)? The "
         "answer is", "{answer}"),
        ("You will be given a text below. Complete the text.\n{sentence}",
         "{answer}"),
        ("TEXT: {sentence}", "{answer}"),
        ("SENTENCE: {sentence}", "{answer}"),
        ("Complete: {sentence}", "{answer}"),
        ("Text complete: {sentence}", "{answer}"),
        ("Complete text: {sentence}", "{answer}"),
        ("Continue writing the following text: {sentence}", "{answer}"),
    ],
    # Not in FLAN Templates (flan_templates):
    "cot_gsm8k": [
        ("{question} Let's think first. Chain of thought:",
         "{chain_of_thought}\nTherefore, the answer is {answer}."),
        ("{question} Think carefully first, then make a decision:",
         "{chain_of_thought} So, the answer is {answer}."),
        ("{question} Let's be accurate as possible.",
         "{chain_of_thought}\nThe answer: {answer}."),
        ("{question} Give me reasons, before answering the question",
         "{chain_of_thought} So the final answer is {answer}."),
        ("Lizzy: {question}.\nMe: Hmmm, let me think. I think this is the "
         "detailed solution:", "{chain_of_thought} Final answer: {answer}."),
        ("Question: {question} Think carefully first, then make a decision:",
         "{chain_of_thought} So the answer is {answer}."),
        ("Give the step-by-step reasoning process and then the final answer. "
         "{question}", "{chain_of_thought}\nThe final answer: {answer}."),
        ("{question}\nThoughts? Step-by-step reasoning:",
         "{chain_of_thought}\nThus, the answer is {answer}."),
        ("My question is: {question} Your thoughts:",
         "{chain_of_thought} The final answer: {answer}."),
        ("{question} Let's answer step by step:",
         "{chain_of_thought} The answer: {answer}."),
    ],
    # Not in FLAN Templates (flan_templates):
    "cot_strategyqa": [
        ("{question}\nThink slowly and carefully, before giving your answer.",
         "{chain_of_thought}\nSo, the answer is {answer}."),
        ("{question} Please answer step by step:",
         "{chain_of_thought}\nSo, the final answer is {answer}."),
        ("{question}\nChain of thought:",
         "{chain_of_thought} The answer is {answer}."),
        ("Answer the following question by reasoning step-by-step. {question}",
         "{chain_of_thought} Therefore, the final answer is {answer}."),
        ("{question} Given the above question, please answer with reasoning "
         "first!", "{chain_of_thought}\nTherefore, the answer is {answer}."),
        ("{question} Think carefully first, then make a decision:",
         "{chain_of_thought} So, the answer is {answer}."),
        ("Q: {question} Now, let's think step by step:",
         "{chain_of_thought}\nThe answer: {answer}."),
        ("Answer the following question, but give the rationale first. "
         "{question}", "{chain_of_thought} So the final answer is {answer}."),
        ("{question} Hmmm, my chain of thoughts:",
         "{chain_of_thought} Final answer: {answer}."),
        ("Let's answer this question slowly: {question}\n",
         "{chain_of_thought} So the answer is {answer}."),
    ],
    # Not in FLAN Templates (flan_templates):
    "cot_creak": [
        ("Given the following question, let's solve step-by-step. {question}\n",
         "{chain_of_thought}\nThe final answer: {answer}."),
        ("My question: {question}\nPlease think gradually:",
         "{chain_of_thought}\nThus, the answer is {answer}."),
        ("Give the rationale and then the answer. {question}",
         "{chain_of_thought} The final answer: {answer}."),
        ("Q: {question}\nChain-of-thought:",
         "{chain_of_thought} The answer: {answer}."),
        ("{question}\nChain of thought and solution for this question is:",
         "{chain_of_thought}\nSo, the answer is {answer}."),
        ("Question: {question} Let's think first. Step-by-step reasoning:",
         "{chain_of_thought}\nSo, the final answer is {answer}."),
        ("{question}\nYour chain-of-thought:",
         "{chain_of_thought} The answer is {answer}."),
        ("{question} Step-by-step reasoning process:",
         "{chain_of_thought} Therefore, the final answer is {answer}."),
        ("{question} The thought process:",
         "{chain_of_thought}\nTherefore, the answer is {answer}."),
        ("{question} Let's think first. Step-by-step reasoning process:",
         "{chain_of_thought} So, the answer is {answer}."),
    ],
    # Not in FLAN Templates (flan_templates):
    "cot_qasc": [
        ("Question: {question}\nLet's be accurate as possible and think "
         "step-by-step.", "{chain_of_thought}\nThe answer: {answer}."),
        ("{question} Let's solve this problem gradually.\n",
         "{chain_of_thought} So the final answer is {answer}."),
        ("Question to you: {question}.\nLet's reason step-by-step:",
         "{chain_of_thought} Final answer: {answer}."),
        ("{question} Think carefully first, then make a decision. My thoughts:",
         "{chain_of_thought} So the answer is {answer}."),
        ("{question} Let's be accurate as possible.",
         "{chain_of_thought}\nThe final answer: {answer}."),
        ("Q: {question}\nLet's think step by step below.\n",
         "{chain_of_thought}\nThus, the answer is {answer}."),
        ("Let's think step by step! {question}\nThe thinking starts now:",
         "{chain_of_thought} The final answer: {answer}."),
        ("{question}\nHmmm, let me think. I don't want to be wrong, so I got "
         "to be careful.", "{chain_of_thought} The answer: {answer}."),
        ("Use reasoning to answer the following question. {question}",
         "{chain_of_thought}\nSo, the answer is {answer}."),
        ("{question} OK. Let's think hard:",
         "{chain_of_thought}\nSo, the final answer is {answer}."),
    ],
    # Not in FLAN Templates (flan_templates):
    "cot_esnli": [
        ("{question}\nLet's solve step-by-step:",
         "{chain_of_thought} The answer is {answer}."),
        ("{question} Step by step answer:",
         "{chain_of_thought} Therefore, the final answer is {answer}."),
        ("{question} Stream of thoughts:",
         "{chain_of_thought}\nTherefore, the answer is {answer}."),
        ("{question} Now, let's be accurate as possible. Some thinking first:",
         "{chain_of_thought} So, the answer is {answer}."),
        ("Denny asked: {question}.\nLe: OK, so how can I answer with some "
         "explanation?\n", "{chain_of_thought}\nThe answer: {answer}."),
        ("Student: {question}.\nTeacher: Let's think:\n",
         "{chain_of_thought} So the final answer is {answer}."),
        ("{question} Let's be accurate as possible and think first.",
         "{chain_of_thought} Final answer: {answer}."),
        ("Please answer the following question by reasoning step-by-step. "
         "{question}. Step-by-step reasoning:",
         "{chain_of_thought} So the answer is {answer}."),
        ("{question} A step-by-step solution is:\n",
         "{chain_of_thought}\nThe final answer: {answer}."),
        ("Leo: {question}\nMei: OK, So, let's think first...\nMe:",
         "{chain_of_thought}\nThus, the answer is {answer}."),
    ],
    # Not in FLAN Templates (flan_templates):
    "cot_ecqa": [
        ("{question}\nPlease answer and provide answer explanation.",
         "{chain_of_thought} The final answer: {answer}."),
        ("{question}\nStep-by-step reasoning process below:\n",
         "{chain_of_thought} The answer: {answer}."),
        ("{question} Hmmm, let me think.",
         "{chain_of_thought}\nSo, the answer is {answer}."),
        ("{question}\nLet's think now! Step-by-step reasoning:",
         "{chain_of_thought}\nSo, the final answer is {answer}."),
        ("next question: {question}\nreasoning:",
         "{chain_of_thought} The answer is {answer}."),
        ("Use reasoning to lead to the answer of the following question:\n"
         "{question}\n Reasoning process:",
         "{chain_of_thought} Therefore, the final answer is {answer}."),
        ("{question} Let's give stream of consciousness first:",
         "{chain_of_thought}\nTherefore, the answer is {answer}."),
        ("{question} Let's think step by step:",
         "{chain_of_thought} So, the answer is {answer}."),
        ("I'll give you a question, please answer with step-by-step reasoning "
         "process. {question}\n", "{chain_of_thought}\nThe answer: {answer}."),
        ("{question}\nLet's think carefully first. Step-by-step reasoning "
         "process:", "{chain_of_thought} So the final answer is {answer}."),
    ],
    # Not in FLAN Templates (flan_templates):
    "cot_sensemaking": [
        ("{question} Let's reason step by step:",
         "{chain_of_thought} Final answer: {answer}."),
        ("Question: {question}\nPlease answer this question gradually:",
         "{chain_of_thought} So the answer is {answer}."),
        ("See question below:\n{question}\nReason slowly and give your answer.",
         "{chain_of_thought}\nThe final answer: {answer}."),
        ("OK. You'll be given the following question. Please do "
         "chain-of-thought reasoning.\n{question}",
         "{chain_of_thought}\nThus, the answer is {answer}."),
        ("{question} Let's be accurate as possible. So think first.",
         "{chain_of_thought}\nThe final answer: {answer}."),
        ("Q: {question}\nLet's solve this gradually.\n",
         "{chain_of_thought} The answer is {answer}."),
        ("Let's think step by step! {question}\n",
         "{chain_of_thought} The answer: {answer}."),
        ("{question}\nHmmm, let me think. I want to lay out the solution "
         "in details.", "{chain_of_thought} The answer is {answer}."),
        ("Answer the following question, with explanation first. {question}",
         "{chain_of_thought}\nSo, the answer is {answer}."),
        ("{question} Let me think hard. Detailed solution:",
         "{chain_of_thought}\nThe answer is {answer}."),
    ],
    # Not in FLAN Templates (flan_templates):
    "stream_aqua": [
        ("Q: {question} Let's give some random thoughts before answering.",
         "{chain_of_thought}\nTherefore, the answer is {answer}."),
        ("{question} Hmmm, my stream of consciousness:",
         "{chain_of_thought} So, the answer is {answer}."),
        ("Give a quick stream of consciousness before answering the following "
         "question. {question}", "{chain_of_thought}\nThe answer: {answer}."),
        ("Use some thinking to answer the following question. {question}",
         "{chain_of_thought} So the final answer is {answer}."),
        ("Student: {question}.\nAnother student: Let's say, hmmm...\n",
         "{chain_of_thought} Final answer: {answer}."),
        ("{question} Think first, then make a decision. Some random thoughts:",
         "{chain_of_thought} So the answer is {answer}."),
        ("{question} Now, let's think a bit. Some random thoughts:",
         "{chain_of_thought}\nThe final answer: {answer}."),
        ("{question} Stream of consciousness:",
         "{chain_of_thought}\nThus, the answer is {answer}."),
        ("Question: {question} Random thoughts:",
         "{chain_of_thought} The final answer: {answer}."),
        ("{question} OK. Let's think. Some random thoughts first:",
         "{chain_of_thought} The answer: {answer}."),
        ("Give stream of consciousness and then the final answer. {question}",
         "{chain_of_thought}\nThe final answer: {answer}."),
        ("{question} Stream of consciousness first, then make a decision:",
         "{chain_of_thought}\nThus, the answer is {answer}."),
        ("Question: {question} Let's think first. Some random reasoning:",
         "{chain_of_thought} The final answer: {answer}."),
        ("Some question: {question}\nSome stream of consciousness:",
         "{chain_of_thought} The answer: {answer}."),
        ("{question} Let's think first. Stream of consciousness:",
         "{chain_of_thought}\nSo, the answer is {answer}."),
    ],
    # Not in FLAN Templates (flan_templates):
    "stream_qed": [
        ("{question}\nSteam of consciousness below:\n",
         "{chain_of_thought}\nSo, the answer is {answer}."),
        ("{question} Let's give stream of consciousness first:",
         "{chain_of_thought}\nSo, the final answer is {answer}."),
        ("Quoc: {question}\nHW Chung: OK, some thoughts:",
         "{chain_of_thought} The answer is {answer}."),
        ("Q: {question} Let's give stream of consciousness first:",
         "{chain_of_thought} Therefore, the final answer is {answer}."),
        ("I got a question for you: {question}\nLet's think first:",
         "{chain_of_thought}\nTherefore, the answer is {answer}."),
        ("{question} Okie... think carefully first, then make a decision:",
         "{chain_of_thought} So, the answer is {answer}."),
        ("Output a stream of consciousness before answering the following. "
         "{question}", "{chain_of_thought}\nThe answer: {answer}."),
        ("{question} Let's think fast. Stream of consciousness:",
         "{chain_of_thought} So the final answer is {answer}."),
        ("Use stream of consciousness to answer the following. {question}",
         "{chain_of_thought} Final answer: {answer}."),
        ("Q: {question}\nLet's give stream of consciousness below\n",
         "{chain_of_thought} So the answer is {answer}."),
        ("Give a stream of consciousness and then the final answer. {question}",
         "{chain_of_thought}\nSo, the final answer is {answer}."),
        ("{question} OK. Let's think. My stream of consciousness:",
         "{chain_of_thought} The answer is {answer}."),
        ("Answer the following Q with stream of consciousness. {question}",
         "{chain_of_thought} Therefore, the final answer is {answer}."),
        ("Give some stream of consciousness and then the answer. {question}",
         "{chain_of_thought}\nTherefore, the answer is {answer}."),
        ("{question} Let's have some stream of consciousness first.",
         "{chain_of_thought} So, the answer is {answer}."),
    ],
    # Not in FLAN Templates (flan_templates):
    "strategyqa": [
        ("Yes or no: {question}", "{answer}"),
        ("{question} Answer yes or no.", "{answer}"),
        ("Question: {question} Answer:", "{answer}"),
        ("Answer yes or no after the question mark: {question}", "{answer}"),
        ("Answer yes or no: {question}", "{answer}"),
        ("Reply yes or no: {question}", "{answer}"),
        ("{question} Yes or no:", "{answer}"),
        ("{question}\n\nIt's yes or no? The answer is", "{answer}"),
        ("Yes/no: {question}", "{answer}"),
        ("{question}. The answer:", "{answer}"),
    ],
    # Not in FLAN Templates (flan_templates):
    "unified_qa_science_inst": [
        ("{question}\n{options_}", "{answer}"),
        ("{question}\n{options_}The answer is:", "{answer}"),
        ("{question} {options_}\nYour answer:", "{answer}"),
        ("{question}\n{options_}\n", "{answer}"),
        ("{question}\n-\n{options_}", "{answer}"),
        ("{question}\n{options_}\n", "{answer}"),
        ("{question} {options_}\n", "{answer}"),
        ("Answer this:\n{question}\n{options_}", "{answer}"),
        ("{question}\n\n{options_}\n\n", "{answer}"),
        ("Answer this question: {question}\n{options_}. Answer:", "{answer}"),
    ],
    # Not in FLAN Templates (flan_templates):
    "bigbench:simple_arithmetic_json.gen.blueridge_vocab.0_shot.30_examples": [
        ("What is the value of {inputs}? Answer:", "{targets}"),
        ("What is the solution of the following "
         "problem?\n{inputs}\n\nSolution:", "{targets}"),
        ("Reply with the result of this math problem:\n\n{inputs}",
         "{targets}"),
        ("{inputs} The answer is", "{targets}"),
        ("Solve this math problem: {inputs}\n\n", "{targets}"),
        ("{inputs}\n\n", "{targets}"),
        ("{inputs} A:", "{targets}"),
        ("Q: {inputs} A:", "{targets}"),
        ("Question: {inputs}\nAnswer:", "{targets}"),
        ("Math problem: {inputs}\nAnswer:", "{targets}"),
    ],
    # Not in FLAN Templates (flan_templates):
    "bigbench:auto_debugging.gen.blueridge_vocab.0_shot.34_examples": [
        ("Answer the following question:\n{inputs}", "{targets}"),
        ("Given the question below, answer directly after the question "
         "ended:\n{inputs}", "{targets}"),
        ("{inputs} I think the answer is:", "{targets}"),
        ("{inputs} The answer is", "{targets}"),
        ("{inputs} The answer of this coding problem is", "{targets}"),
        ("{inputs} Hmm... The answer is", "{targets}"),
        ("{inputs} Hmm... I believe the correct answer should be", "{targets}"),
        ("Answer the following coding question:\n\n{inputs}\n\n", "{targets}"),
        ("See this interesting question:\n{inputs}\nThe quick answer is:",
         "{targets}"),
        ("{inputs} A:", "{targets}"),
    ],
    # Not in FLAN Templates (flan_templates):
    "bigbench:strategyqa.gen.blueridge_vocab.0_shot.1000_examples": [
        ("{inputs} First answer yes or no, then explain.", "{targets}"),
        ("Answer this question (yes or no) then explain why:\n{inputs}",
         "{targets}"),
        ("Yes or no: {inputs} The answer followed by explanation is:",
         "{targets}"),
        ("Answer yes or no after the question mark, then explain the reason: "
         "{inputs}", "{targets}"),
        ("Yes or no first, then explain the reason: {inputs}", "{targets}"),
        ("Yes/no: {inputs}", "{targets}"),
        ("You will be given a question. Answer yes or no first, then give the "
         "reason.\n{inputs}\n\n", "{targets}"),
        ("{inputs} Answer followed by reasoning:", "{targets}"),
        ("Answer + your thought for the following question: {inputs}\n",
         "{targets}"),
        ("{inputs} Answer + thought is:", "{targets}"),
    ],
    # Not in FLAN Templates (flan_templates):
    "bigbench:sufficient_information.gen.blueridge_vocab.0_shot.39_examples": [
        ("{inputs}\n\n", "{targets}"),
        ("Answer this question or say \"I don't know\": {inputs}", "{targets}"),
        ("Q: {inputs} A:", "{targets}"),
        ("Answer the given question (if the question cannot be answered due to"
         " lack of information, answer \"I don't know\").\n{inputs}",
         "{targets}"),
        ("Question: {inputs}\n\nAnswer:", "{targets}"),
        ("Q: {inputs}\nA:", "{targets}"),
        ("The answer (if no enough information, say I don't know) to "
         "\"{inputs}\" is:", "{targets}"),
        ("Question that might not be answerable: {inputs}. Answer:",
         "{targets}"),
        ("Question: {inputs}\nAnswer:", "{targets}"),
        ("{inputs} A:", "{targets}"),
    ],
    # Not in FLAN Templates (flan_templates):
    "predict_next_turn_dialog": [
        ("{dialog_}", "{answer}"),
        ("{dialog_}\n", "{answer}"),
        ("Read the dialog and predict the next turn. {dialog_}\n", "{answer}"),
        ("What is the next dialog turn? {dialog_}", "{answer}"),
        ("See the conversation. {dialog_}", "{answer}"),
        ("Write the response. {dialog_}", "{answer}"),
        ("Write the conversation response. {dialog_}", "{answer}"),
        ("Fill in the response. {dialog_}", "{answer}"),
        ("What was likely said next? {dialog_}", "{answer}"),
        ("What was the response? {dialog_}", "{answer}"),
    ],
    # Not in FLAN Templates (flan_templates):
    "t0_question_answer": [
        # t0 comes pre-templatized/formatted and generation task varies
        # e.g. QA or question generation
        ("{question}\n", "{answer}"),
        ("{question}\nAnswer:", "{answer}"),
        ("{question}\nA:", "{answer}"),
        ("Q:{question}\nA:", "{answer}"),
        ("Question: {question}\nAnswer:", "{answer}"),
        ("Answer the following question: {question}\nAnswer:", "{answer}"),
        ("Given the question: {question}\nThe answer is:", "{answer}"),
        ("{question}\nThe answer to this question is:", "{answer}"),
        ("Please answer the following question: {question}\nA:", "{answer}"),
        ("Please answer the following question: {question}\nAnswer:",
         "{answer}"),
    ],
    # Not in FLAN Templates (flan_templates):
    "t0_multiple_choice_separated_options": [
        ("{question}\n{options_}", "{answer}"),
        ("{question}\n{options_}\nAnswer:", "{answer}"),
        ("{question}\n\n{options_}\nAnswer:", "{answer}"),
        ("Q: {question}\n\n{options_}\nA:", "{answer}"),
        ("Answer the following question: {question}\n\n{options_}\nAnswer:",
         "{answer}"),
        ("{options_}\n\n{question}\nAnswer:", "{answer}"),
        ("{options_}\nQ: {question}\nA:", "{answer}"),
        ("{question}\n\n{options_}\nThe answer is:", "{answer}"),
        ("{options_}\nGiven those answer options, answer the "
         "question: {question}\nA:", "{answer}"),
        ("Q: {question}\n\n{options_}\nThe answer is:", "{answer}"),
    ],
    # Not in FLAN Templates (flan_templates):
    "program_synthesis_dmcc_python": [
        ("{question}", "{answer}"),
        ("Write a program that answers the question. {question}\nAnswer:",
         "{answer}"),
        ("Write code that solves this problem. {question}\nAnswer:",
         "{answer}"),
        ("Write a program that solves this problem. {question}\nSolution:",
         "{answer}"),
        ("Solve this problem. {question}\nSolution:", "{answer}"),
        ("Solve this problem. {question}\nSolution in code:", "{answer}"),
        ("{question}\n\nCode solution:", "{answer}"),
        ("Coding Problem.\n{question}\n\nSolution:", "{answer}"),
        ("{question}\n\nCode solution in Python:", "{answer}"),
        ("[code]{question}[BEGIN]", "{answer}[DONE]"),
    ],
    # Not in FLAN Templates (flan_templates):
    "program_synthesis_dr_repair": [
        ("My broken code is below:\n{question}\nThe fixed code should be:",
         "{answer}"),
        ("My broken code is below:\n{question}\nThe fixed code:", "{answer}"),
        ("Incorrect code:\n{question}\nFixed code:", "{answer}"),
        ("Incorrect code:\n{question}\n\nThe correct version:", "{answer}"),
        ("This code is broken:\n{question}\n\nShow the fixed version:",
         "{answer}"),
        ("Broken:\n{question}\n\nFixed:", "{answer}"),
        ("Broken code:\n{question}\n\nFixed Code:", "{answer}"),
        ("The following code is not correct.\n{question}\n\nPropose solution "
         "code:", "{answer}"),
        ("The following code is not correct.\n{question}\n\nCome up with code "
         "that would fix this:", "{answer}"),
        ("Fix this code. ```{question}```\n\nA potential fix:```",
         "{answer}```"),
    ],
    # Not in FLAN Templates (flan_templates):
    "program_synthesis_dr_repair_error_comments": [
        ("My broken code is below with errors in comments:\n{question}\nThe "
         "fixed code should be:", "{answer}"),
        ("My broken code is below with errors in comments:\n{question}\nThe "
         "fixed code, with no more errors or error comments:", "{answer}"),
        ("Errors are described inline in comments. Incorrect "
         "code:\n{question}\nFixed code:", "{answer}"),
        ("See errors in comments. Incorrect code:\n{question}\n\nThe correct "
         "version:", "{answer}"),
        ("This code is broken:\n{question}\n\nVersion which fixes commented "
         "errors:", "{answer}"),
        ("Broken:\n```{question}```\n\nFixed:```", "{answer}```"),
        ("Broken code (see error comments):\n{question}\n\nFixed:", "{answer}"),
        ("Coding Challenge: fix the errors, as commented:\n{question}\n\nFixed:",
         "{answer}"),
        ("Challenge Question. See code:\n{question}\n\nA potential fix:",
         "{answer}"),
        ("Fix this code. ```{question}```\n\nA potential fix:```",
         "{answer}```"),
    ],
    # Not in FLAN Templates (flan_templates):
    "cot_stream_general_input_inversion": [
        # CoT + Answer --> Question
        ("Given the following reasoning and answer, what was the question? "
         "{chain_of_thought}\n The answer: {answer}", "The question {question}"
        ),
        # CoT + Answer --> Question
        ("For this chain-of-thought reasoning and answer, what was the "
         "question?\n{chain_of_thought}\n A: {answer}", "Q: {question}"),
        # Question + Answer --> CoT
        ("Consider the question. {question}\n What is the step-by-step "
         "reasoning process to arrive at the answer: {answer}?",
         "{chain_of_thought}"),
        # Question + Answer --> CoT
        ("Question. {question}\nAnswer. {answer}\nWhat step-by-step "
         "reasoning justifies that answer?", "Reasoning: {chain_of_thought}"),
        # Question + Answer --> CoT
        ("Q: {question}\nA: {answer}\nExplain how we arrive at this answer: ",
         "Explanation: {chain_of_thought}"),
        # CoT --> Question + Answer
        ("Given the rationale, provide a reasonable question and answer. "
         "Step-by-step reasoning process: {chain_of_thought}\n The question "
         "and answer:", "{question}\nThe answer is {answer}"),
        # CoT --> Question + Answer
        ("{chain_of_thought}\nThis justifies what answer for what question? Q "
         "& A: ", "{question}\n{answer}"),
        # CoT --> Question + Answer
        ("{chain_of_thought}is the reasoning for what question and answer pair?",
         "Q: {question}\nA: {answer}"),
        # Answer --> Question + CoT
        ("Come up with a question and reasoning that would justify this "
         "answer: {answer}", "The question is: {question}\n"
         "Step-by-step reasoning process: {chain_of_thought}\n"),
        # Answer --> Question + CoT
        ("Creatively image a question and justification for this answer: "
         "{answer}", "The question is: {question}\nStep-by-step reasoning "
         "process: {chain_of_thought}\n"),
        # CoT + Answer --> Question
        ("What was the question for this implicit rationale, and corresponding"
         " answer?\n{chain_of_thought}\n The answer: {answer}",
         "The question: {question}"),
        # Question + Answer --> CoT
        ("Consider the question. {question}\n If the answer is '{answer}'; "
         "explain the reasoning:", "{chain_of_thought}"),
        # Question + Answer --> CoT
        ("Explain simply why {answer} is the correct answer to: {question}. "
         "Explanation:", "{chain_of_thought}"),
        # CoT --> Question + Answer
        ("Given the stream of consciousness rationale, provide a reasonable "
         "question and answer. Rationale: {chain_of_thought}\n The question "
         "and answer:", "{question}\nThe answer is {answer}"),
        # CoT --> Question + Answer
        ("Stream of consciousness rationale: {chain_of_thought}\nThe question "
         "and answer pair are described below.", "Q: {question}\nA: {answer}"),
        # CoT --> Question + Answer
        ("Reconstruct a question, answer pair from this explanation: "
         "{chain_of_thought}\n", "Q:{question}\nA:{answer}"),
        # Answer --> Question + CoT
        ("Come up with a question and stream of consciousness reasoning that "
         "would justify this answer: {answer}", "The question is: {question}\n"
         "Stream of consciousness: {chain_of_thought}\n"),
        # Answer --> Question + CoT
        ("Imagine a question and stream-of-consciousness explanation for which"
         " this is the answer: {answer}", "Question: {question}\n"
         "Stream-of-consciousness: {chain_of_thought}"),
    ],
    # Not in FLAN Templates (flan_templates):
    "predict_next_turn_dialog_input_inversion": [
        ("Consider this response: {answer}\nWhat was the preceding dialog?",
         "{dialog_}"),
        ("{answer}\nThe preceding conversation:", "{dialog_}"),
        ("Read this response and predict the preceding dialog. {answer}\n",
         "{dialog_}"),
        ("What might have been said before this? {answer}", "{dialog_}"),
        ("{answer}\nPrevious conversation:", "{dialog_}"),
        ("What came before. {answer}", "{dialog_}"),
        ("Write the conversation that led to this response. {answer}",
         "{dialog_}"),
        ("See this dialog response. {answer} What came before?", "{dialog_}"),
        ("Imagine the conversation that came before this response? Response: "
         "{answer}", "{dialog_}"),
        ("If this is the response, what came before? Response {answer}",
         "{dialog_}"),
    ],
    # Not in FLAN Templates (flan_templates):
    "program_synthesis_dmcc_python_input_inversion": [
        ("If this is the answer: {answer}\n what was the question?",
         "{question}"),
        ("This program answers a question. {answer}\nQuestion:", "{question}"),
        ("Write a problem which this code solves. {answer}\nProblem:",
         "{question}"),
        ("This program is the solution to a question. {answer}\nQuestion:",
         "{question}"),
        ("Solution: {answer}\nThe corresponding question:", "{question}"),
        ("Solution code: {answer}\nThe problem:", "{question}"),
        ("Code solution: {answer}\n\nProblem this solves:", "{question}"),
        ("Coding Problem. Solution:\n{answer}\n\nQuestion:", "{question}"),
        ("Code solution in Python: {answer}\n\nSolves this question:",
         "{question}"),
        ("[BEGIN]{answer}[DONE]\nCode Problem:", "{question}"),
    ],
    # Not in FLAN Templates (flan_templates):
    "program_synthesis_dr_repair_input_inversion": [
        ("Fixed code: {answer}.\nMy broken code is below:\n", "{question}"),
        ("The fixed code:\n{answer}\nMy broken code is below:", "{question}"),
        ("Fixed code:\n{answer}\nExample of incorrect code:", "{answer}"),
        ("Correct version of the code:\n{answer}\n\nCode with error:",
         "{question}"),
        ("This is the solution code:\n{answer}\n\nWhich fixes this version:",
         "{question}"),
        ("Fixed:\n{answer}\n\nBroken:", "{question}"),
        ("Fixed code:\n{answer}\n\nBroken Code:", "{question}"),
        ("The following code is the solution.\n{answer}\n\nPropose an "
         "incorrect solution. HERE: ", "{question}"),
        ("The following code is correct.\n{answer}\n\nCome up with error code "
         "that this fixes:", "{question}"),
        ("A potential fix here: ```{answer}```\n\nBroken version:```",
         "{question}```"),
    ],
    # Not in FLAN Templates (flan_templates):
    # NB: ALL NatInstV2 tasks come somewhat pre-templatized.
    "natinst_v2": [
        ("{Definition}\n\n{input}", "{output}"),
        ("You will be given a definition of a task first, then some input of "
         "the task.\n{Definition}\n\n{input}\nOutput:", "{output}"),
        ("Definition: {Definition}\nInput: {input}\nOutput:", "{output}"),
        ("Instructions: {Definition}\nInput: {input}\nOutput:", "{output}"),
        ("{Definition}\nQ: {input}\nA: ", "{output}"),
        ("Given the task definition and input, reply with output. "
         "{Definition}\n\n{input}\n", "{output}"),
        ("Teacher:{Definition}\nTeacher: Now, understand the problem? Solve "
         "this instance: {input}\nStudent:", "{output}"),
        ("Q: {Definition}\n{input}\nA:", "{output}"),
        ("Detailed Instructions: {Definition}\nProblem:{input}\nSolution:",
         "{output}"),
        ("Detailed Instructions: {Definition}\nQ: {input}\nA:", "{output}"),
    ],
}

PATTERNS["wiki_dialog"] = PATTERNS["predict_next_turn_dialog"]
PATTERNS["task_master"] = PATTERNS["predict_next_turn_dialog"]
PATTERNS["qrecc"] = PATTERNS["predict_next_turn_dialog"]
PATTERNS["wiki_dialog_input_inversion"] = PATTERNS[
    "predict_next_turn_dialog_input_inversion"]
PATTERNS["task_master_input_inversion"] = PATTERNS[
    "predict_next_turn_dialog_input_inversion"]
PATTERNS["qrecc_input_inversion"] = PATTERNS[
    "predict_next_turn_dialog_input_inversion"]

PATTERNS["t0_multiple_choice"] = PATTERNS["t0_question_answer"]

PATTERNS["program_synthesis_dr_repair_no_errors"] = PATTERNS[
    "program_synthesis_dr_repair"]
PATTERNS["program_synthesis_dr_repair_plain_code"] = PATTERNS[
    "program_synthesis_dr_repair"]
PATTERNS["program_synthesis_dr_repair_line_numbers"] = PATTERNS[
    "program_synthesis_dr_repair"]

PATTERNS["program_synthesis_dr_repair_no_errors_input_inversion"] = PATTERNS[
    "program_synthesis_dr_repair_input_inversion"]
PATTERNS["program_synthesis_dr_repair_plain_code_input_inversion"] = PATTERNS[
    "program_synthesis_dr_repair_input_inversion"]
PATTERNS["program_synthesis_dr_repair_line_numbers_input_inversion"] = PATTERNS[
    "program_synthesis_dr_repair_input_inversion"]
PATTERNS[
    "program_synthesis_dr_repair_error_comments_input_inversion"] = PATTERNS[
        "program_synthesis_dr_repair_input_inversion"]

PATTERNS["cot_input_inversion_gsm8k"] = PATTERNS[
    "cot_stream_general_input_inversion"]
PATTERNS["cot_input_inversion_strategyqa"] = PATTERNS[
    "cot_stream_general_input_inversion"]
PATTERNS["cot_input_inversion_creak"] = PATTERNS[
    "cot_stream_general_input_inversion"]
PATTERNS["cot_input_inversion_qasc"] = PATTERNS[
    "cot_stream_general_input_inversion"]
PATTERNS["cot_input_inversion_esnli"] = PATTERNS[
    "cot_stream_general_input_inversion"]
PATTERNS["cot_input_inversion_ecqa"] = PATTERNS[
    "cot_stream_general_input_inversion"]
PATTERNS["cot_input_inversion_sensemaking"] = PATTERNS[
    "cot_stream_general_input_inversion"]
PATTERNS["stream_input_inversion_aqua"] = PATTERNS[
    "cot_stream_general_input_inversion"]
PATTERNS["stream_input_inversion_qed"] = PATTERNS[
    "cot_stream_general_input_inversion"]

# Here we create train templates *without* any answer options.
# FLAN templates did not indicate there were options, it just presented them.
PATTERNS_NO_OPTIONS = copy.deepcopy(PATTERNS)
PATTERNS_NO_OPTIONS.update(flan_templates.PATTERNS)
for t_name, templates in PATTERNS_NO_OPTIONS.items():
  for ti, (in_template, out_template) in enumerate(templates):
    PATTERNS_NO_OPTIONS[t_name][ti] = (in_template.replace(
        "\n\n{options_}",
        "").replace("\n{options_}",
                    "").replace("{options_}",
                                "").replace("\n\n{options_str}",
                                            ""), out_template)


# Few-shot patterns.
@dataclasses.dataclass
class FewShotPattern:
  """Patterns for few-shot tasks.

  The few-shot input are composed by a few examplers followed by final_suffix:
  {exampler no. 1} + {exampler no. 2} + {exampler no. 3}... + {final_suffix}

  Each exampler has the following format:
  {inputs_prefix} + {inputs} + {x_y_delimiter} + {targets_prefix} + {targets} +
  {example_separator}
  """
  inputs: str
  targets: str
  inputs_prefix: str = ""
  targets_prefix: str = ""
  x_y_delimiter: str = "\n\n"
  example_separator: str = "\n\n\n"
  final_suffix: str = ""
  input_pattern: str = "{{inputs}}{final_suffix}"
  in_template_mix: bool = True

  @property
  def few_shot_kwargs(self):
    return dict(
        inputs_prefix=self.inputs_prefix,
        targets_prefix=self.targets_prefix,
        x_y_delimiter=self.x_y_delimiter,
        example_separator=self.example_separator,
        final_suffix=self.final_suffix,
        input_pattern=self.input_pattern)

  @property
  def combined_inputs(self):
    return self.inputs_prefix + self.inputs + self.x_y_delimiter

  @property
  def combined_targets(self):
    return self.targets_prefix + self.targets + self.example_separator

  @property
  def combined_inputs_w_target_prefix(self):
    return self.inputs_prefix + self.inputs + self.x_y_delimiter + (
        self.targets_prefix)

  @property
  def combined_targets_wo_target_prefix(self):
    return self.targets + self.example_separator


FEWSHOT_PATTERNS = {
    "rte": [
        FewShotPattern(
            inputs="Context: {premise}\n{hypothesis}\n{options_}",
            targets="{answer}",
            inputs_prefix="Problem: ",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{premise}\n{hypothesis}\n{options_}",
            targets="{answer}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{premise}\nCan we say the following?\n{hypothesis}"
            "\n{options_}",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{premise}\n\n{hypothesis}\n\n{options_}",
            targets="{answer}",
            inputs_prefix="Question: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n",
            example_separator="\n"),
        FewShotPattern(
            inputs="Context: {premise}\n\nGenerate a hypothesis.",
            targets="Hypothesis: {hypothesis}",
            inputs_prefix="Question: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n****\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Determine if the sentence is true based on the text below.\n{hypothesis}\n\n{premise}\n{options_}",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n"),
        FewShotPattern(
            inputs="{premise}\n{options_}\nQuestion: {hypothesis}",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{premise}\nIs this true: {hypothesis}\n{options_}",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="A: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{premise}\nIs this true?\n{hypothesis}\n{options_}",
            targets="{answer}",
            inputs_prefix="Question:\n",
            targets_prefix="Answer:\n",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{premise}\n\nBased on that paragraph can we say the following?\n{hypothesis}\n\n{options_}",
            targets="{answer}",
            inputs_prefix="Problem: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n****\n",
            example_separator="\n\n\n"),
    ],
    "wsc": [
        FewShotPattern(
            inputs="{context}\n{text1}\n{text2}\n{options_}",
            targets="{answer}",
            inputs_prefix="Question:\n",
            targets_prefix="Answer:\n",
            x_y_delimiter="\n-----\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{context}\nAre \"{text1}\" and \"{text2}\" the same? "
            "{options_}",
            targets="{answer}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n=======\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Given context:{context}\n\n1: {text1}; 2: {text2}"
            "\n{options_}",
            targets="{answer}",
            inputs_prefix="Question: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n",
            example_separator="\n"),
        FewShotPattern(
            inputs="{context}\n\nDo \"{text2}\" and \"{text1}\" mean the same "
            "thing? {options_}",
            targets="{answer}",
            inputs_prefix="Problem: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Do \"{text1}\" and \"{text2}\" point to the same thing in the following sentence?\n\n{context}\n\n{options_}",
            targets="{answer}",
            inputs_prefix="Problem: ",
            targets_prefix="A: ",
            x_y_delimiter="\n****\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Is \"{text1}\" the same as \"{text2}\" in this sentence?\n{context}\n\n{options_}",
            targets="{answer}",
            inputs_prefix="QUESTION: ",
            targets_prefix="ANS: ",
            x_y_delimiter="\n",
            example_separator="\n"),
        FewShotPattern(
            inputs="{context}\nAre \"{text2}\" and \"{text1}\" the same?\n{options_}",
            targets="{answer}",
            inputs_prefix="Problem: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n****\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{context}\n\nAre \"{text2}\" and \"{text1}\" the same?\n\n{options_}",
            targets="{answer}",
            inputs_prefix="Problem: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{options_}.\n\n{context}\n\nDo \"{text2}\" and \"{text1}\" mean the same thing?",
            targets="{answer}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n"),
        FewShotPattern(
            inputs="CONTEXT: {context}\n\nMulti-choice question: Do \"{text1}\" and \"{text2}\" have the same meaning?\n\n{options_}",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n\n"),
    ],
    "wsc273": [
        FewShotPattern(
            inputs="{context} {options_}",
            targets="{answer}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Complete the passage.\n\n{context}\n{options_}",
            targets="{answer}",
            inputs_prefix="Problem: ",
            targets_prefix="A: ",
            x_y_delimiter="\n  **  \n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="How does this following sentence end?\n\n{context}"
            "\n{options_}",
            targets="{answer}",
            inputs_prefix="Question:\n",
            targets_prefix="Answer:\n",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="What is the most logical completion for the following text?"
            "\n{context}\n{options_}",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n"),
        FewShotPattern(
            inputs="Complete:\n\n{context}\n{options_}",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="A: ",
            x_y_delimiter="\n**\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Fill in the remainder of the sentence.\n\n{context}\n{options_}",
            targets="{answer}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Complete the following.\n{context}\n{options_}",
            targets="{answer}",
            inputs_prefix="Problem: ",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n"),
        FewShotPattern(
            inputs="Choose the text that follows: {context}\n{options_}",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="So how does this text end?\n\n{context}\n{options_}",
            targets="{answer}",
            inputs_prefix="question: ",
            targets_prefix="answer: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{context}\n{options_}",
            targets="{answer}",
            inputs_prefix="Question: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
    ],
    "wic": [
        FewShotPattern(
            inputs="{sentence1}\n{sentence2}\nWord: \"{word}\"\n{options_}",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Sentence 1: {sentence1}\nSentence 2: {sentence2}\nDoes "
            "{word} mean the same thing in these two sentences?\n{options_}",
            targets="{answer}",
            inputs_prefix="Question: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n============\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Sentence1: {sentence1}\nSentence2: {sentence2}\nQ: Does "
            "the term {word} mean the same thing in both these sentences?"
            "\n{options_}",
            targets="{answer}",
            inputs_prefix="Problem: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Sentence (1) {sentence1}\nSentence (2) {sentence2}. "
            "Word: {word}. {options_}",
            targets="{answer}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n==--==\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Does the word \"{word}\" mean the same thing in \"{sentence1}\" and \"{sentence2}\"?\n{options_}",
            targets="{answer}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Word: {word}\n\nSentence 1: {sentence1}\n\nSentence 2: {sentence2}\n\nSame meaning? {options_}",
            targets="{answer}",
            inputs_prefix="Problem: ",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="This question has options. Does the word \"{word}\" have the same definition in the next 2 sentences?\n\n{sentence1}\n\n{sentence2}\n\n{options_}",
            targets="{answer}",
            inputs_prefix="Question: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="\"{word}\" used in the same way in the following two sentences?\n\n{sentence1}\n\n{sentence2}\n\n{options_}",
            targets="{answer}",
            inputs_prefix="Question: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n",
            example_separator="\n"),
        FewShotPattern(
            inputs="Does \"{word}\" have the same meaning in the following two sentences?\n\n{sentence1}\n\n{sentence2}\n\n{options_}",
            targets="{answer}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Here is one sentence: {sentence1}\nHere is another sentence: {sentence2}\nDoes the {word} mean the same thing in the two sentences?\n{options_}",
            targets="{answer}",
            inputs_prefix="Question: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
    ],
    "record": [
        FewShotPattern(
            inputs="Complete:\n{passage}\n{query}\n{options_str}",
            targets="{answer}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{passage}\n\n{query}\n\n{options_str}",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n=====\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Find the right ending to this passage."
            "\n\n{passage}\n-->\n{query}\n\n{options_str}",
            targets="{answer}",
            inputs_prefix="Problem: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Passage: {passage}\nQuery: {query}\n{options_str}",
            targets="{answer}",
            inputs_prefix="Question: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n",
            example_separator="\n"),
        FewShotPattern(
            inputs="{passage}\n\n{query}\n\n{options_str}",
            targets="{answer}",
            inputs_prefix="Problem: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Let's complete this passage.\n{passage}\n\n{query}\n\n{options_str}",
            targets="{answer}",
            inputs_prefix="Question: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n",
            example_separator="\n"),
        FewShotPattern(
            inputs="Choose the next sentence\n{passage}\n\n{query}\n\n{options_str}",
            targets="{answer}",
            inputs_prefix="Problem: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="A COMPLETION PROBLEM. \n\n{passage}\n\n{query}\n\n{options_str}",
            targets="{answer}",
            inputs_prefix="QUES: ",
            targets_prefix="ANS: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="[passage completion]\n\n{passage}\n{query}\n{options_str}",
            targets="{answer}",
            inputs_prefix="Question:\n",
            targets_prefix="Answer:\n",
            x_y_delimiter="\n\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Choose the next sentence.\n{passage}\n{query}\n{options_str}",
            targets="{answer}",
            inputs_prefix="Input: ",
            targets_prefix="Output: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
    ],
    "natural_questions": [
        FewShotPattern(
            inputs="{question}??",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="Answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{question}",
            targets="{answer}",
            inputs_prefix="Question: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Answer this question: {question}?",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{question}",
            targets="{answer}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{question}",
            targets="{answer}",
            inputs_prefix="Input: ",
            targets_prefix="Output: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="answer this: {question}?",
            targets="{answer}",
            inputs_prefix="QUESTION: ",
            targets_prefix="ANS: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="What is the answer to this question? {question}",
            targets="{answer}",
            inputs_prefix="input: ",
            targets_prefix="output: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{question}?",
            targets="{answer}",
            inputs_prefix="In: ",
            targets_prefix="Out: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Please answer this: {question}",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Answer this question:\n\n{question}?",
            targets="{answer}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
    ],
    "synth_cot_natural_questions": [
        FewShotPattern(
            inputs="{question}??",
            targets="Answer: {cot}. So the answer is {answer}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="Chain-of-thought: {cot}. Answer: {answer}",
            inputs_prefix="Question: ",
            targets_prefix="",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Answer this question: {question}?",
            targets="{cot}. [{answer}]",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="{cot}. So... {answer}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{question}?",
            targets="logic: {cot}\n\n{answer}",
            inputs_prefix="",
            targets_prefix="Answer: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}\nLet's think...",
            targets="{cot}. So {answer} is the answer.",
            inputs_prefix="QUESTION: ",
            targets_prefix="ANS: ",
            x_y_delimiter="\n****\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="{answer}\nExplanation: {cot}",
            inputs_prefix="",
            targets_prefix="Answer: ",
            x_y_delimiter="\n***\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Please answer this question: {question}\nGive your reasons first.",
            targets="{cot}\n\nAnswer is {answer}",
            inputs_prefix="in: ",
            targets_prefix="out: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Answer this question:\n\n{question}? Think out loud!",
            targets="{cot}\nSo, the answer is {answer}",
            inputs_prefix="Student A: ",
            targets_prefix="Student B: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Answer this:\n\n{question}",
            targets="{cot}\nThe answer is {answer}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n",
            in_template_mix=False),
    ],
    "trivia_qa": [
        FewShotPattern(
            inputs="{question}",
            targets="{answer}",
            inputs_prefix="Problem: ",
            targets_prefix="A: ",
            x_y_delimiter="\n======\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{question}",
            targets="{answer}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{question}",
            targets="{answer}",
            inputs_prefix="Problem: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{question}",
            targets="{answer}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n-----\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{question}\nWhat is the answer?",
            targets="{answer}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Trivia question: {question}",
            targets="{answer}",
            inputs_prefix="QUESTION: ",
            targets_prefix="ANS: ",
            x_y_delimiter="\n****\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{question}?",
            targets="{answer}",
            inputs_prefix="Mei Li:\n",
            targets_prefix="Shuai Zheng:\n",
            x_y_delimiter="\n",
            example_separator="\n"),
        FewShotPattern(
            inputs="{question}",
            targets="{answer}",
            inputs_prefix="Input question: ",
            targets_prefix="Output answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Answer this question.\n\n{question}",
            targets="{answer}",
            inputs_prefix="question: ",
            targets_prefix="answer: ",
            x_y_delimiter="\n",
            example_separator="\n"),
        FewShotPattern(
            inputs="What is the answer: {question}",
            targets="{answer}",
            inputs_prefix="question: ",
            targets_prefix="answer: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
    ],
    "math_dataset": [
        FewShotPattern(
            inputs="{question}",
            targets="{answer}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Solve:\n{question}",
            targets="{answer}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="What is the solution?\n\n{question}",
            targets="{answer}",
            inputs_prefix="Question: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Math Problem\n{question}",
            targets="{answer}",
            inputs_prefix="Problem: ",
            targets_prefix="A: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{question}.",
            targets="{answer}",
            inputs_prefix="Question:\n",
            targets_prefix="Answer:\n",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Teacher asked me this: {question}",
            targets="{answer}",
            inputs_prefix="question: ",
            targets_prefix="answer: ",
            x_y_delimiter="\n++++++++++\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{question}\nSolve this plz.",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="A: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Math problem: {question}",
            targets="{answer}",
            inputs_prefix="QUESTION: ",
            targets_prefix="ANS: ",
            x_y_delimiter="\n",
            example_separator="\n"),
        FewShotPattern(
            inputs="What is the solution?\n{question}",
            targets="{answer}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Write down the solution for this math problem: {question}",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="answer: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
    ],
    "aeslc": [
        FewShotPattern(
            inputs="{body}",
            targets="{subject}",
            inputs_prefix="",
            targets_prefix="Answer: ",
            x_y_delimiter="\n",
            example_separator="\n"),
        FewShotPattern(
            inputs="Here is an email: {body}\nWhat is a potential subject line "
            "for this email?",
            targets="{subject}",
            inputs_prefix="",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n"),
        FewShotPattern(
            inputs="{body}\nPropose a subject line",
            targets="{subject}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="BODY: {body}",
            targets="SUBJECT: {subject}",
            inputs_prefix="Problem: ",
            targets_prefix="A: ",
            x_y_delimiter="\n======\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="\"{subject}\".",
            targets="{body}",
            inputs_prefix="Subject: ",
            targets_prefix="Body: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{body}\n\nGenerate a subject line.",
            targets="{subject}",
            inputs_prefix="IN: ",
            targets_prefix="OUT: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="This is an email\n{body}\n\nWhat is the subject?",
            targets="{subject}",
            inputs_prefix="IN: ",
            targets_prefix="OUT: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="This is the content of an email: {body}",
            targets="Subject: {subject}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{body}\nPropose a subject line for this email?",
            targets="{subject}",
            inputs_prefix="QUESTION: ",
            targets_prefix="ANS: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{body}\nQ: What is a potential subject line for this email?",
            targets="{subject}",
            inputs_prefix="",
            targets_prefix="answer: ",
            x_y_delimiter="\n",
            example_separator="\n"),
    ],
    "cnn_dailymail": [
        FewShotPattern(
            inputs="{text}",
            targets="{highlights}",
            inputs_prefix="question: ",
            targets_prefix="answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{text}",
            targets="{highlights}",
            inputs_prefix="Text: ",
            targets_prefix="Highlights: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="TEXT: {text}",
            targets="Highlights: {highlights}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{text}\nWhat are the important parts of this article?",
            targets="{highlights}",
            inputs_prefix="Question: ",
            targets_prefix="Important parts: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{highlights}",
            targets="{text}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n****\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{text}",
            targets="{highlights}",
            inputs_prefix="in: ",
            targets_prefix="out: ",
            x_y_delimiter="\n",
            example_separator="\n"),
        FewShotPattern(
            inputs="{text}\nSummarize this article.",
            targets="{highlights}",
            inputs_prefix="Problem: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{text}\n\nWhat are highlight points?",
            targets="{highlights}",
            inputs_prefix="Problem: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{text}\n\nWrite highlights.",
            targets="{highlights}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{highlights}",
            targets="{text}",
            inputs_prefix="Input: ",
            targets_prefix="Output: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
    ],
    "gigaword": [
        FewShotPattern(
            inputs="{text}",
            targets="{summary}",
            inputs_prefix="",
            targets_prefix="A summary about the text above: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Summarize this: {text}",
            targets="{summary}",
            inputs_prefix="Question:\n",
            targets_prefix="Answer:\n",
            x_y_delimiter="\n",
            example_separator="\n"),
        FewShotPattern(
            inputs="{text}",
            targets="{summary}",
            inputs_prefix="Text: ",
            targets_prefix="Summary: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{text}",
            targets="{summary}",
            inputs_prefix="Generate a short summary: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{text}",
            targets="{summary}",
            inputs_prefix="",
            targets_prefix="Short summary: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{text}",
            targets="{summary}",
            inputs_prefix="IN: ",
            targets_prefix="summary: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{text}\n\nWhat is a very short summary of the above text?",
            targets="{summary}",
            inputs_prefix="Problem: ",
            targets_prefix="A: ",
            x_y_delimiter="\n++++++++++\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{text}\n\nWrite a brief summary in a sentence or so.",
            targets="{summary}",
            inputs_prefix="question: ",
            targets_prefix="summary: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Write a text based on \"{summary}\"",
            targets="{text}",
            inputs_prefix="Question: ",
            targets_prefix="Text: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Write something based on this summary: {summary}\n\nSentence:",
            targets="{text}",
            inputs_prefix="",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n",
            in_template_mix=False),
    ],
    "multi_news": [
        FewShotPattern(
            inputs="Article:\n\n{text}\n",
            targets="{summary}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{text}",
            targets="{summary}",
            inputs_prefix="Text:\n",
            targets_prefix="Summary:\n",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Here is a news article: {text}\nA summary of this is?",
            targets="{summary}",
            inputs_prefix="Question:\n",
            targets_prefix="Answer:\n",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="News article:\n\n{text}\nWhat is a shorter version of the "
            "above article?",
            targets="{summary}",
            inputs_prefix="question: ",
            targets_prefix="Shorter version: ",
            x_y_delimiter="\n------\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{summary}",
            targets="Let's expand this into a news article: {text}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{text}",
            targets="{summary}",
            inputs_prefix="input: ",
            targets_prefix="summary: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{text}",
            targets="{summary}",
            inputs_prefix="Article:\n",
            targets_prefix="In short:\n",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Here is a news article: {text}",
            targets="Here is a summary: {summary}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{text}\nWhat is a one-paragraph summary of the above article?",
            targets="{summary}",
            inputs_prefix="input question: ",
            targets_prefix="output answer: ",
            x_y_delimiter="\n",
            example_separator="\n"),
        FewShotPattern(
            inputs="Article:\n\n{text}\nWhat is a summary?",
            targets="{summary}",
            inputs_prefix="Q: ",
            targets_prefix="A summary: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
    ],
    "newsroom": [
        FewShotPattern(
            inputs="{title}\n\n{text}",
            targets="{summary}",
            inputs_prefix="",
            targets_prefix="summary: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{title}\n{text}\nWhat was this article about?",
            targets="{summary}",
            inputs_prefix="Q:\n",
            targets_prefix="A:\n",
            x_y_delimiter="\n",
            example_separator="\n"),
        FewShotPattern(
            inputs="{title}\n{text}\nPlease write a summary below.",
            targets="{summary}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n"),
        FewShotPattern(
            inputs="{title}\n\n{text}\nWhat are the most important parts of "
            "this text?",
            targets="{summary}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{text}",
            targets="{title}",
            inputs_prefix="News: ",
            targets_prefix="Title: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{title}\n\n{text}",
            targets="{summary}",
            inputs_prefix="",
            targets_prefix="Summary: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{title}\n\n{text}",
            targets="{summary}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{title}\n{text}\nWhat is a short summary of the above article?",
            targets="{summary}",
            inputs_prefix="IN: ",
            targets_prefix="OUT: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{title}\n{text}\nWhat was this article about?",
            targets="{summary}",
            inputs_prefix="Input: ",
            targets_prefix="It's about: ",
            x_y_delimiter="\n++++++++++\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{title}\n\n{text}\n\nWrite a one or two sentence summary.",
            targets="{summary}",
            inputs_prefix="Problem: ",
            targets_prefix="Summary: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
    ],
    "samsum": [
        FewShotPattern(
            inputs="{dialogue}\n\nBriefly summarize:",
            targets="{summary}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Here is a dialogue:\n{dialogue}\n\nWrite a short summary!",
            targets="{summary}",
            inputs_prefix="",
            targets_prefix="summary: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Dialogue:\n{dialogue}",
            targets="{summary}",
            inputs_prefix="Question: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n",
            example_separator="\n"),
        FewShotPattern(
            inputs="{dialogue}\n\nWhat was that dialogue about?",
            targets="{summary}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Write a dialog with this premise \"{summary}\".",
            targets="{dialogue}",
            inputs_prefix="question: ",
            targets_prefix="dialog: ",
            x_y_delimiter="\n",
            example_separator="\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Write a dialog based on this summary:\n{summary}.",
            targets="{dialogue}",
            inputs_prefix="input question: ",
            targets_prefix="dialogue: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Write a random dialog about: {summary}.",
            targets="{dialogue}",
            inputs_prefix="Input: ",
            targets_prefix="Output: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Dialogue:\n{dialogue}\nWhat was going on?",
            targets="{summary}",
            inputs_prefix="IN: ",
            targets_prefix="OUT: ",
            x_y_delimiter="\n****\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{dialogue}",
            targets="{summary}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{dialogue}",
            targets="{summary}",
            inputs_prefix="Part A --- ",
            targets_prefix="Part B --- ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
    ],
    "xsum": [
        FewShotPattern(
            inputs="{text}",
            targets="{summary}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{text}",
            targets="{summary}",
            inputs_prefix="Text: ",
            targets_prefix="summary: ",
            x_y_delimiter="\n",
            example_separator="\n"),
        FewShotPattern(
            inputs="Article: {text}",
            targets="[[{summary}]]",
            inputs_prefix="Problem: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Article:{text}\n\nSummarize the main points of that "
            "article.",
            targets="{summary}",
            inputs_prefix="Question:\n",
            targets_prefix="Answer:\n",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Article: {text}\n\nSummarize.",
            targets="{summary}",
            inputs_prefix="input: ",
            targets_prefix="output: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Article: {text}",
            targets="{summary}",
            inputs_prefix="",
            targets_prefix="Summarize: ",
            x_y_delimiter="\n++++++++++\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{text}\nWhat was that article about?",
            targets="{summary}",
            inputs_prefix="Problem: ",
            targets_prefix="A: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{text}",
            targets="{summary}",
            inputs_prefix="",
            targets_prefix="Sum: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Write an article based on this \"{summary}\"\n\nArticle:",
            targets="{text}",
            inputs_prefix="Problem: ",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Write an article based on this summary:\n\n{summary}",
            targets="{text}",
            inputs_prefix="",
            targets_prefix="A: ",
            x_y_delimiter="\n+++++++\n",
            example_separator="\n\n\n",
            in_template_mix=False),
    ],
    "squad_v1": [
        FewShotPattern(
            inputs="Please answer a question about the following article about "
            "{title}:\n{context}\n{question}",
            targets="{answer}",
            inputs_prefix="Problem: ",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n"),
        FewShotPattern(
            inputs="Read this and answer the question\n\n{context}"
            "\n\n{question}",
            targets="{answer}",
            inputs_prefix="Question: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n",
            example_separator="\n"),
        FewShotPattern(
            inputs="{context}\n{question}",
            targets="{answer}",
            inputs_prefix="Problem: ",
            targets_prefix="The answer is the following: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{context}\n{question}",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{title}.",
            targets="{question}",
            inputs_prefix="Title: ",
            targets_prefix="Question: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{title}\n{context}\n\n{question}",
            targets="{answer}",
            inputs_prefix="Input: ",
            targets_prefix="Output: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Article: {context}\n\nNow answer this question: {question}",
            targets="{answer}",
            inputs_prefix="Input: ",
            targets_prefix="Output: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Article: {context}\n\nQuestion: {question}",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="Ans: ",
            x_y_delimiter="\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Here is a question about this article: {context}\nWhat is the answer to this question: {question}",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="So... ",
            x_y_delimiter="\n****\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Answer a question about this article:\n{context}\n{question}",
            targets="{answer}",
            inputs_prefix="The problem: ",
            targets_prefix="The answer: ",
            x_y_delimiter="\n****\n",
            example_separator="\n\n\n"),
    ],
    "squad_v2": [
        FewShotPattern(
            inputs="{title}:\n\n{context}\n\n{question}",
            targets="{answer}",
            inputs_prefix="Problem: ",
            targets_prefix="A: ",
            x_y_delimiter="\n---\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{context}\n{question}",
            targets="{answer}",
            inputs_prefix="Problem: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n---\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="What is a question about this article? If the question is "
            "unanswerable, say \"unanswerable\".\n{context}\n{question}",
            targets="{answer}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{context}\n{question}",
            targets="{answer}",
            inputs_prefix="Context and question: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n",
            example_separator="\n"),
        FewShotPattern(
            inputs="{context}\nIs there an answer to this question: {question}",
            targets="{answer}",
            inputs_prefix="Question: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Read this: {context}\nQuestion: {question}",
            targets="{answer}",
            inputs_prefix="Input: ",
            targets_prefix="Output: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{context}\n\n{question}\nWhat is the answer?",
            targets="{answer}",
            inputs_prefix="QUES: ",
            targets_prefix="ANS: ",
            x_y_delimiter="\n",
            example_separator="\n"),
        FewShotPattern(
            inputs="{context}\n{question}",
            targets="{answer}",
            inputs_prefix="QUES: ",
            targets_prefix="ANS: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{context}\nIf it is possible to answer this question, answer it for me (else, reply \"unanswerable\"): {question}",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="Ah, so.. ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{context}\nTry to answer this question if possible: {question}",
            targets="{answer}",
            inputs_prefix="Question: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n",
            example_separator="\n"),
    ],
    "drop": [
        FewShotPattern(
            inputs="{context}\n{question}",
            targets="{answer}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{context}\nAnswer this question based on the article: "
            "{question}",
            targets="{answer}",
            inputs_prefix="Problem: ",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{context}\n\n{question}",
            targets="{answer}",
            inputs_prefix="Question:\n",
            targets_prefix="Answer:\n",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{context}\nAnswer this question: {question}",
            targets="{answer}",
            inputs_prefix="question: ",
            targets_prefix="answer: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{context}",
            targets="{question}",
            inputs_prefix="Input: ",
            targets_prefix="Question: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Write an article that answers the following question: {question}",
            targets="Article: {context}",
            inputs_prefix="Question: ",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{context}\n\n{question}",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{context}\n{question}",
            targets="{answer}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{context}\nAnswer this: {question}",
            targets="{answer}",
            inputs_prefix="P: ",
            targets_prefix="A: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{context}\n\n{question}",
            targets="{answer}",
            inputs_prefix="Problem: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
    ],
    "synth_cot_drop": [
        FewShotPattern(
            inputs="{context}\n{question}",
            targets="{cot} The answer is {answer}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{context}\nAnswer this question based on the article: "
            "{question}",
            targets="{cot} The answer is {answer}",
            inputs_prefix="Problem: ",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{context}\n\n{question}",
            targets="{cot} The answer is {answer}",
            inputs_prefix="Question:\n",
            targets_prefix="I think:\n",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{context}\nAnswer this question: {question}",
            targets="{cot} The answer is {answer}",
            inputs_prefix="question: ",
            targets_prefix="Chain-of-thought: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{context} {question}",
            targets="{cot} The answer is {answer}",
            inputs_prefix="Question:\n",
            targets_prefix="Answer:\n",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{context}\n\nBased on the above article, answer a question. {question}",
            targets="{cot}\nANS: {answer}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{context}\n{question}",
            targets="The answer is {answer}.\nExplanation: {cot}",
            inputs_prefix="QUES: ",
            targets_prefix="ANS: ",
            x_y_delimiter="\n++++++++++\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{context}\n{question}. Now, let me think...",
            targets="{cot}\nSo, I would say the answer to this question is {answer}",
            inputs_prefix="Question: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n",
            example_separator="\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{context}\n\nAnswer this question: {question}",
            targets="{cot} The answer is {answer}",
            inputs_prefix="",
            targets_prefix="A: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{context} {question} Chain-of-thought:",
            targets="{cot} {answer}",
            inputs_prefix="[Q]: ",
            targets_prefix="[A]: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
    ],
    "quac": [
        FewShotPattern(
            inputs="{background}\n{context}\n{question}",
            targets="{answer}",
            inputs_prefix="Some context: ",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n"),
        FewShotPattern(
            inputs="{background}\n\n{context}\n\nUsing a quote from the above "
            "article, answer the following question: {question}",
            targets="{answer}",
            inputs_prefix="Question: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\nHHHHHH\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{background}\n\n{context}\n\n{question}",
            targets="{answer}",
            inputs_prefix="Problem: ",
            targets_prefix="Answer with quotes: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Background: {background}\nContext: {context}\n"
            "Question: {question}",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="Answer: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{background}\n\n{context}",
            targets="{question}",
            inputs_prefix="IN: ",
            targets_prefix="QUESTION: ",
            x_y_delimiter="\n",
            example_separator="\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{background}\n{context}\nAnswer this question using a quote from the text above:\n\n{question}",
            targets="{answer}",
            inputs_prefix="Question:\n",
            targets_prefix="Answer:\n",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{context}\n\nAnswer this question \"{question}\"",
            targets="{answer}",
            inputs_prefix="input: ",
            targets_prefix="output: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Background: {background}\nContext: {context}\nQuestion: {question}",
            targets="{answer}",
            inputs_prefix="Problem: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{background}\n\n{context}\n\n{question}",
            targets="{answer}",
            inputs_prefix="IN: ",
            targets_prefix="OUT: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Answer the question at the end by quoting:\n\n{background}\n{context}\n\n{question}",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n\n"),
    ],
    "para_crawl": [
        FewShotPattern(
            inputs="How do you say \"{sent1}\" in {lang2}?",
            targets="{sent2}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{sent2} ** {lang1}?",
            targets="{sent1}",
            inputs_prefix="Problem: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="1: {sent1}; 2: {lang2}.",
            targets="{sent2}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Translate \"{sent2}\": {lang2} --> {lang1}.",
            targets="{sent1}",
            inputs_prefix="Q: ",
            targets_prefix="translate: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{lang2}.",
            targets="{sent2}",
            inputs_prefix="Problem: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Language: {lang1}.",
            targets="Sentence: {sent1}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Translate \"{sent1}\" to {lang2}.",
            targets="{sent2}",
            inputs_prefix="[Translate Q]: ",
            targets_prefix="[A]: ",
            x_y_delimiter="\n",
            example_separator="\n"),
        FewShotPattern(
            inputs="Translate \"{sent2}\" from {lang2} to {lang1}.",
            targets="{sent1}",
            inputs_prefix="Question: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{sent1} Say this using {lang2}.",
            targets="{sent2}",
            inputs_prefix="Question: ",
            targets_prefix="Say: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{sent2} How do you say this sentence in {lang1}?",
            targets="{sent1}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n"),
    ],
    "wmt16_translate": [
        FewShotPattern(
            inputs="{sent2}\n{lang1}?",
            targets="{sent1}",
            inputs_prefix="test: ",
            targets_prefix="translation: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{sent2}\n\nIn {lang1}?",
            targets="{sent1}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\nxxxxx\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="How is \"{sent1}\" said in {lang2}?",
            targets="{sent2}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Translate \"{sent1}\" to {lang2}?",
            targets="{sent2}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{sent2}\n\nWhich language is this?",
            targets="{lang2}",
            inputs_prefix="",
            targets_prefix="Language: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Write a sentence not in {lang1}.",
            targets="{sent2}",
            inputs_prefix="input question: ",
            targets_prefix="output answer: ",
            x_y_delimiter="\n",
            example_separator="\n"),
        FewShotPattern(
            inputs="Translate \"{sent1}\" to {lang2}?",
            targets="{sent2}",
            inputs_prefix="Q: ",
            targets_prefix="Yes: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="How is \"{sent1}\" said in {lang2}?",
            targets="In {lang2}: {sent2}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{sent2}\n\nTranslate this to {lang1}?",
            targets="{sent1}",
            inputs_prefix="[Q]: ",
            targets_prefix="[A]: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{sent2}\n\nCould you please translate this to {lang1}?",
            targets="{sent1}",
            inputs_prefix="Question:\n",
            targets_prefix="Answer:\n",
            x_y_delimiter="\n",
            example_separator="\n\n"),
    ],
    "wmt14_enfr": [
        FewShotPattern(
            inputs="{sent1} --> {lang2}.",
            targets="{sent2}",
            inputs_prefix="question: ",
            targets_prefix="answer: ",
            x_y_delimiter="\n",
            example_separator="\n"),
        FewShotPattern(
            inputs="{sent2}\n\nTranslate to {lang1}.",
            targets="{sent1}",
            inputs_prefix="Some text: ",
            targets_prefix="Translation: ",
            x_y_delimiter="\n",
            example_separator="\n"),
        FewShotPattern(
            inputs="{sent2}\n\nCould you please translate this to {lang1}?",
            targets="{sent1}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{sent2}... {lang1}?",
            targets="[{sent1}]",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{sent2}\n\nWhich language is this?",
            targets="{lang2}",
            inputs_prefix="Problem: ",
            targets_prefix="Language: ",
            x_y_delimiter="\n+++++\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Write a sentence not in {lang1}.",
            targets="{sent2}",
            inputs_prefix="Question:\n",
            targets_prefix="Sentence:\n",
            x_y_delimiter="\n",
            example_separator="\n-+-+-+-\n"),
        FewShotPattern(
            inputs="\"{sent1}\" --> {lang2}?",
            targets="{sent2}",
            inputs_prefix="[Q]: ",
            targets_prefix="[A]: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="\"{sent1}\" to {lang2}",
            targets="{sent2}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n****\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Translate to {lang2}:\n{sent1}",
            targets="{sent2}",
            inputs_prefix="",
            targets_prefix="Answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{lang2}:\n\n{sent1}",
            targets="{sent2}",
            inputs_prefix="IN: ",
            targets_prefix="OUT: ",
            x_y_delimiter="\n",
            example_separator="\n"),
    ],
    "true_case": [
        FewShotPattern(
            inputs="{lower}\n\nPlease write the text above using proper case.",
            targets="**{answer}**",
            inputs_prefix="",
            targets_prefix="A: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{lower}",
            targets="--> {answer}",
            inputs_prefix="Question: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{lower}\n\nHow would the previous sentence be correctly "
            "capitalized?",
            targets="Answer: {answer}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{lower}\nCapitalize this past sentence correctly.",
            targets="A: {answer}",
            inputs_prefix="question: ",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{answer}",
            targets="{lower}",
            inputs_prefix="IN: ",
            targets_prefix="OUT: ",
            x_y_delimiter="\n**\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Please capitalize where necessary: {lower}",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{lower}",
            targets="{answer}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n\n",
            example_separator="\n++++++++\n"),
        FewShotPattern(
            inputs="{lower}",
            targets="{answer}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{lower}\nCan you repeat this sentence, but capitalize it?",
            targets="{answer}",
            inputs_prefix="Problem: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n",
            example_separator="\n"),
        FewShotPattern(
            inputs="{lower}",
            targets="{answer}",
            inputs_prefix="Text: ",
            targets_prefix="Correct capitalization: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
    ],
    "fix_punct": [
        FewShotPattern(
            inputs="{no_punct}",
            targets="{answer}",
            inputs_prefix="Question: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n--\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{no_punct}\n\nCan you repeat this sentence, but add in "
            "punctuation?",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n"),
        FewShotPattern(
            inputs="{no_punct}",
            targets="A: {answer}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{no_punct}\nFix the punctuation.",
            targets="Ans: {answer}",
            inputs_prefix="Ques: ",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{no_punct}",
            targets="{answer}",
            inputs_prefix="input: ",
            targets_prefix="fixed: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{no_punct}",
            targets="{answer}",
            inputs_prefix="input ---- ",
            targets_prefix="output ---- ",
            x_y_delimiter="\n",
            example_separator="\n"),
        FewShotPattern(
            inputs="Add punctuation: {no_punct}",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Fix punctuation: {no_punct}",
            targets="{answer}",
            inputs_prefix="QUESTION: ",
            targets_prefix="ANS: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{no_punct}\n\ncorrect the punctuation.",
            targets="{answer}",
            inputs_prefix="QUES: ",
            targets_prefix="CORRECTED: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{no_punct}\nPlease fix the punctuation.",
            targets="[{answer}]",
            inputs_prefix="Question: ",
            targets_prefix="Fixed: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
    ],
    "word_segment": [
        FewShotPattern(
            inputs="{no_space}\nWhat's a sentence that uses these characters?",
            targets="{answer}",
            inputs_prefix="Problem: ",
            targets_prefix="",
            x_y_delimiter="\n-\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Add spaces: {no_space}",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="Answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{no_space}",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="This text is missing some spaces, please add them: "
            "{no_space}",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Remove the spaces from the following sentence: {answer}",
            targets="[{no_space}]",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{answer}\n\nPlease remove spaces.",
            targets="{no_space}",
            inputs_prefix="Question: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="[{no_space}]",
            targets="[{answer}]",
            inputs_prefix="[Q]: ",
            targets_prefix="[A]: ",
            x_y_delimiter="\n",
            example_separator="\n"),
        FewShotPattern(
            inputs="Add spaces: {no_space}",
            targets="{answer}",
            inputs_prefix="Problem: ",
            targets_prefix="Spaces added: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="This text is missing some spaces, please add them: {no_space}",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n"),
        FewShotPattern(
            inputs="Fix spacing: {no_space}",
            targets="{answer}",
            inputs_prefix="input question: ",
            targets_prefix="output answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
    ],
    "cosmos_qa": [
        FewShotPattern(
            inputs="{context} {options_}",
            targets="{question}",
            inputs_prefix="",
            targets_prefix="A: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{context}\n\n{question}\n{options_}",
            targets="{answer}",
            inputs_prefix="Problem: ",
            targets_prefix="A: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{context} {options_}",
            targets="Answer: {answer}",
            inputs_prefix="Question: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{context}\nAnswer the following question: {question}"
            "\n{options_}",
            targets="{answer}",
            inputs_prefix="Question:\n",
            targets_prefix="Answer:\n",
            x_y_delimiter="\n",
            example_separator="\n"),
        FewShotPattern(
            inputs="{context}",
            targets="{question}\n{options_}",
            inputs_prefix="Context: ",
            targets_prefix="Question generated: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{context}\n\n{question}\n{options_}",
            targets="{answer}",
            inputs_prefix="Question:\n",
            targets_prefix="Answer:\n",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{context}\n{question}\n{options_}",
            targets="{answer}",
            inputs_prefix="IN: ",
            targets_prefix="OUT: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{context}\n\n{options_}\nAnswer the following question: {question}",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{context}\n\n{options_}\nQ: {question}",
            targets="{answer}",
            inputs_prefix="question: ",
            targets_prefix="answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{context}\n\nQuestion with options: {question}\n{options_}",
            targets="{answer}",
            inputs_prefix="QUES: ",
            targets_prefix="ANS: ",
            x_y_delimiter="\n",
            example_separator="\n"),
    ],
    "synth_cot_cosmos_qa": [
        FewShotPattern(
            inputs="{context}\n{question}\n{options_}",
            targets="{cot} The answer is {answer}",
            inputs_prefix="",
            targets_prefix="A: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{context}\n\n{question}\n{options_}",
            targets="{cot} The answer is {answer}",
            inputs_prefix="Problem: ",
            targets_prefix="A: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{context}\n{question}\n{options_}",
            targets="Chain-of-thought: {cot}\nSo the answer is {answer}",
            inputs_prefix="Question: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{context}\nAnswer the following question: {question}"
            "\n{options_}",
            targets="{cot} The answer is {answer}",
            inputs_prefix="Question:\n",
            targets_prefix="Answer:\n",
            x_y_delimiter="\n",
            example_separator="\n"),
        FewShotPattern(
            inputs="{context}\nSolve the following question thinking out loud: {question}\n{options_}",
            targets="{cot} [{answer}]",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{context}\n\n{options_}\nLet's answer this: {question}",
            targets="{cot}\nThe answer is {answer}",
            inputs_prefix="",
            targets_prefix="Answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{context}\n\nQuestion: {question}\n{options_}\nLet's answer step by step.",
            targets="[{cot}] So the answer is {answer}",
            inputs_prefix="QUES: ",
            targets_prefix="ANS: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Context: {context}\nQ: {question}",
            targets="{cot}\nThe answer is {answer}",
            inputs_prefix="",
            targets_prefix="[A]: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{context}\nQuestion: {question}",
            targets="{cot} The answer is {answer}",
            inputs_prefix="input question: ",
            targets_prefix="output answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{context}\n\n{question}\n{options_}",
            targets="{answer}. {cot}",
            inputs_prefix="question: ",
            targets_prefix="answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n",
            in_template_mix=False),
    ],
    "ag_news_subset": [
        FewShotPattern(
            inputs="{text} {options_}",
            targets="{title}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{title}\n\n{text}\n\n{options_}",
            targets="{answer}",
            inputs_prefix="question: ",
            targets_prefix="answer: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{title}\n\n{text}\n\nWhich topic is this article about?"
            "\n{options_}",
            targets="Answer: {answer}",
            inputs_prefix="Question: ",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{text}\nQ: Which is the best summary of this article?\n"
            "{options_}\nI think the answer is",
            targets="{answer}",
            inputs_prefix="question: ",
            targets_prefix="answer: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{text}\n\nWhat is a good title for this?",
            targets="{title}",
            inputs_prefix="Problem: ",
            targets_prefix="Title: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{text}\n\n{options_}",
            targets="{answer}",
            inputs_prefix="IN: ",
            targets_prefix="OUT: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{text}\n\n{options_}",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="Correct title: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{text}\nWhat's this about?\n\n{options_}",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{text}\n{options_}",
            targets="{answer}",
            inputs_prefix="Input: ",
            targets_prefix="Output: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Choose your answer. {title}\n\n{text}\nWhich topic is this article about?\n{options_}",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
    ],
    "bool_q": [
        FewShotPattern(
            inputs="{text}\nCan we conclude that {question}?\n{options_}",
            targets="{answer}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{text}\n{question}?\n{options_}",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="Answer: ",
            x_y_delimiter="\n+++++\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{text}\n\n{options_}\n\n{question}?",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="A: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Text: {text}\n\nQuestion: {question}?\n\n{options_}",
            targets="{answer}",
            inputs_prefix="Problem: ",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Is it true that {question}?\n\n{text}\n\n{options_}",
            targets="{answer}",
            inputs_prefix="IN: ",
            targets_prefix="OUT: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{title}\n\n{text}\n\n{options_}\n{question}",
            targets="{answer}",
            inputs_prefix="input: ",
            targets_prefix="output: ",
            x_y_delimiter="\n****\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{text} {question}\n\n{options_}",
            targets="{answer}",
            inputs_prefix="Question: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{text}\n{question}?\n\n{options_}",
            targets="{answer}",
            inputs_prefix="IN: ",
            targets_prefix="OUT: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="What's the best answer to this question: {question}?\n\n{options_}",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="Answer: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Text: {text}\n\nQuestion: {question}?\n\n{options_}",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="Answer: ",
            x_y_delimiter="\n[separator]\n",
            example_separator="\n\n\n"),
    ],
    "synth_cot_bool_q": [
        FewShotPattern(
            inputs="{passage}\nCan we say {question}?\n{options_}",
            targets="{cot}. The answer is {answer}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{passage}\n{question}?\n{options_}",
            targets="{cot}. So {answer}",
            inputs_prefix="",
            targets_prefix="Answer: ",
            x_y_delimiter="\n==\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{passage}\n\n{options_}\n\n{question}?",
            targets="{cot}. Answer: {answer}",
            inputs_prefix="",
            targets_prefix="A: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Text: {passage}\n\nQuestion: {question}?\n\n{options_}",
            targets="CoT: {cot}. Ans: {answer}",
            inputs_prefix="Problem: ",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{passage} Is it true that {question}?\n\n{options_}",
            targets="{cot}. The answer is {answer}.",
            inputs_prefix="Problem: ",
            targets_prefix="A: ",
            x_y_delimiter="\n+++++++++\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{passage}\nBased on the above text, what's the best answer? {question}\n{options_}",
            targets="{cot}. Answer: {answer}",
            inputs_prefix="Problem: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}?\n{options_}...",
            targets="{cot}. The answer is [{answer}]",
            inputs_prefix="input question: ",
            targets_prefix="output answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="passage: {passage}\n\nQuestion: {question}?\n\n{options_}",
            targets="Think out loud: {cot}\nThe answer is {answer}",
            inputs_prefix="",
            targets_prefix="Answer: ",
            x_y_delimiter="\n****\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{passage}\n\n{question}?\n\n{options_}\nLet's think step by step.",
            targets="{cot}\nThe answer is {answer}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{passage}\n\nCan we conclude that {question}?\n\n{options_}",
            targets="{cot}. So {answer}",
            inputs_prefix="input question:\n",
            targets_prefix="output answer:\n",
            x_y_delimiter="\n",
            example_separator="\n"),
    ],
    "definite_pronoun_resolution": [
        FewShotPattern(
            inputs="{sentence}\n\n{pronoun}\n{options_}",
            targets="{answer}",
            inputs_prefix="Question:\n",
            targets_prefix="Answer:\n",
            x_y_delimiter="\n",
            example_separator="\n"),
        FewShotPattern(
            inputs="{sentence}\n\nWho is \"{pronoun}\" in this prior sentence?"
            "\n{options_}",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n"),
        FewShotPattern(
            inputs="{sentence}\n\nWho is {pronoun} referring to in this "
            "sentence? {options_}",
            targets="{answer}",
            inputs_prefix="Question:\n",
            targets_prefix="Answer:\n",
            x_y_delimiter="\n---\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{sentence}\nWho {pronoun} is? {options_}",
            targets="{answer}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{sentence}\nWho is \"{pronoun}\"?\n{options_}",
            targets="{answer}",
            inputs_prefix="QUESTION:\n",
            targets_prefix="ANS:\n",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Which person is {pronoun}?\n{sentence}\n\n{options_}",
            targets="{answer}",
            inputs_prefix="IN: ",
            targets_prefix="OUT: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Who is {pronoun}?\n{sentence}\n\n{options_}",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="A:\n",
            x_y_delimiter="\n\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Multi-choice problem: [{pronoun}]\n\n{sentence}\n\n{options_}",
            targets="{answer}",
            inputs_prefix="IN:\n",
            targets_prefix="OUT:\n",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Who is {pronoun} in the following sentence?\n\n{sentence}\n\n{options_}",
            targets="{answer}",
            inputs_prefix="QUES:\n",
            targets_prefix="ANS:\n",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{sentence}\nBased on this sentence, who is {pronoun}?\n\n{options_}",
            targets="{answer}",
            inputs_prefix="Question: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
    ],
    "glue_mrpc": [
        FewShotPattern(
            inputs="Wwo sentences:\n{sentence1}\n{sentence2}\nDo they have the "
            "same meaning? {options_}",
            targets="{answer}",
            inputs_prefix="Problem: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Here are two sentences:\n\n{sentence1}\n\n{sentence2}"
            "\n{options_}",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{sentence1} -- {sentence2} -- {options_}",
            targets="\" {answer} \"",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{sentence1}\n\n{sentence2}\n\nPlease tell me if the "
            "sentences above mean the same.\n{options_}",
            targets="{answer}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n"),
        FewShotPattern(
            inputs="Do these sentences have the same meaning?\n{sentence1}\n{sentence2}\n\n{options_}",
            targets="{answer}",
            inputs_prefix="Q:\n",
            targets_prefix="A:\n",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{sentence1}\n{sentence2}\n\n{options_}",
            targets="{answer}",
            inputs_prefix="input question: ",
            targets_prefix="output answer: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{sentence1}\n{sentence2}\n(See options at the end). If the first sentence is true, is the second one also true?\n{options_}",
            targets="{answer}",
            inputs_prefix="[Q]: ",
            targets_prefix="[A]: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="S1: {sentence1}\nS2: {sentence2}\nAre S1 and S2 the same?\n{options_}",
            targets="{answer}",
            inputs_prefix="Question:\n",
            targets_prefix="Answer:\n",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{sentence1}\n\n{sentence2}\n\nThe sentences above mean the same or not?\n{options_}",
            targets="{answer}",
            inputs_prefix="QUES: ",
            targets_prefix="ANS: ",
            x_y_delimiter="\n****\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{sentence1}\n\n{sentence2}\n\n{options_}",
            targets="{answer}",
            inputs_prefix="[Q]: ",
            targets_prefix="[A]: ",
            x_y_delimiter="\n****\n",
            example_separator="\n\n\n"),
    ],
    "glue_qqp": [
        FewShotPattern(
            inputs="{question1}\n{question2}\n{options_}",
            targets="{answer}",
            inputs_prefix="Problem: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{question1}\n{question2}\n{options_}",
            targets="{answer}",
            inputs_prefix="question: ",
            targets_prefix="answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{question1}\n{question2}\n{options_}",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="Answer: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{question1}\n{question2}\n\nPlease tell me if those "
            "questions are the same. {options_}",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{question1}\n{question2}\n\n{options_}",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="Answer: ",
            x_y_delimiter="\n",
            example_separator="\n"),
        FewShotPattern(
            inputs="Question 1: {question1}\nQuestion 2: {question2}\n{options_}\nWould the answer to these two questions be the same?",
            targets="{answer}",
            inputs_prefix="Question:\n",
            targets_prefix="Answer:\n",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Question 1: {question1}\nQuestion 2: {question2}\n{options_}",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n"),
        FewShotPattern(
            inputs="First question: {question1}\nSecond question: {question2}\nAre these two questions asking the same thing?\n{options_}",
            targets="{answer}",
            inputs_prefix="QUES:\n",
            targets_prefix="ANS:\n",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Q1: {question1}\n\nQ2: {question2}\n\n{options_}",
            targets="{answer}",
            inputs_prefix="Questions: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{question1}\n\n{question2}\n\nPlease tell me if those questions are the same.\n{options_}",
            targets="{answer}",
            inputs_prefix="INPUT: ",
            targets_prefix="SAME OR NOT OUTPUT: ",
            x_y_delimiter="\n****\n",
            example_separator="\n\n\n"),
    ],
    "imdb_reviews": [
        FewShotPattern(
            inputs="Write a {answer} ({options_}) movie review.",
            targets="{text}",
            inputs_prefix="Problem: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{options_} {answer}",
            targets="{text}",
            inputs_prefix="Problem: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n=++++=\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{text} {options_}",
            targets="A: {answer}",
            inputs_prefix="Q: ",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="An example of a movie review. {answer} ({options_}).",
            targets="{text}",
            inputs_prefix="",
            targets_prefix="A: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="What's an example of a movie review?",
            targets="{text}",
            inputs_prefix="Q: ",
            targets_prefix="An example of a movie review is here: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Review: {text}\nWhat is this review like?\n{options_}",
            targets="{answer}",
            inputs_prefix="IN: ",
            targets_prefix="OUT: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Review: {text}\nWhat is it saying??\n{options_}",
            targets="{answer}",
            inputs_prefix="QUESTION: ",
            targets_prefix="ANS: ",
            x_y_delimiter="\n\n",
            example_separator="\n=====\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="This review is like what?\n{options_}\n\nReview:{text}",
            targets="{answer}",
            inputs_prefix="Problem: ",
            targets_prefix="A: ",
            x_y_delimiter="\n+++++++++\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Select the correct sentiment of the following review: {text}\n{options_}",
            targets="{answer}",
            inputs_prefix="input question: ",
            targets_prefix="output answer: ",
            x_y_delimiter="\n....\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{text}\n\nIs the sentiment of this review positive or negative?\n{options_}",
            targets="{answer}",
            inputs_prefix="input: ",
            targets_prefix="pos/neg ans: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            in_template_mix=False),
    ],
    "paws_wiki": [
        FewShotPattern(
            inputs="Please check if these have the same meaning."
            "\n{sentence1}\n{sentence2}\n{options_}",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="Answer: ",
            x_y_delimiter="\n",
            example_separator="\n"),
        FewShotPattern(
            inputs="{sentence1}\n{sentence2}\n{options_}",
            targets="{answer}",
            inputs_prefix="Problem: ",
            targets_prefix="A: ",
            x_y_delimiter="\n-+-\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{sentence1}\n{sentence2}\n\nAre these two sentences "
            "paraphrases of each other? {options_}",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="1. {sentence1}\n2. {sentence2}\n{options_}",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="A same meaning or not problem: {options_}\n{sentence1}\n{sentence2}",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="[A]: ",
            x_y_delimiter="\n****\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Do these mean the same?\n{sentence1}\n{sentence2}\n\n{options_}",
            targets="[{answer}]",
            inputs_prefix="Question:\n",
            targets_prefix="Answer:\n",
            x_y_delimiter="\n****\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Are these paraphrases?\n{sentence1}\n{sentence2}\n\n{options_}",
            targets="[{answer}]",
            inputs_prefix="Input:\n",
            targets_prefix="Output:\n",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Same meaning?\n{sentence1}\n{sentence2}\n\n{options_}",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="Answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="S1: {sentence1}\nS2: {sentence2}\n\nDo S1 & S2 convey the same information?\n\n{options_}",
            targets="{answer}",
            inputs_prefix="Problem: ",
            targets_prefix="A: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="(1) {sentence1}\n(2) {sentence2}\n\nDo (1) and (2) mean the same thing?\n\n{options_}",
            targets="{answer}",
            inputs_prefix="Problem: ",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
    ],
    "synth_cot_paws_wiki": [
        FewShotPattern(
            inputs="Please check if these have the same meaning."
            "\n{sentence1}\n{sentence2}\n{options_}",
            targets="{cot} The answer is {answer}",
            inputs_prefix="",
            targets_prefix="Answer: ",
            x_y_delimiter="\n",
            example_separator="\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{sentence1}\n{sentence2}\n{options_}",
            targets="{cot} {answer}",
            inputs_prefix="Problem: ",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{sentence1}\n{sentence2}\n\nAre these two sentences "
            "paraphrases of each other? {options_} THINK FIRST!",
            targets="{cot} Answer: {answer}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="1. {sentence1}\n2. {sentence2}\n{options_}\nSo...",
            targets="{cot} {answer}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Same-meaning-or-not problem: Answer \"yes\" if they do, otherwise \"no\".\n{sentence1}\n{sentence2}\nYour thought?",
            targets="{cot}\nThe answer is {answer}",
            inputs_prefix="",
            targets_prefix="ANS: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{sentence1}\n{sentence2}\n\n{options_}",
            targets="{cot} {answer}",
            inputs_prefix="IN: ",
            targets_prefix="OUT: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Same meaning?\n{sentence1}\n{sentence2}\n\n{options_}",
            targets="{cot}\nMy answer: {answer}",
            inputs_prefix="Question:\n",
            targets_prefix="Answer:\n",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="S1: {sentence1}\nS2: {sentence2}\n\n{options_}",
            targets="So, let's think. {cot} The answer is {answer}",
            inputs_prefix="",
            targets_prefix="S1 & S2 the same? ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="(1) {sentence1}\n(2) {sentence2}\n\n{options_}",
            targets="{cot}. So [{answer}]",
            inputs_prefix="[Q]: ",
            targets_prefix="[Compare (1) and (2)]: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{sentence1}\n{sentence2}\n\nAre these two sentences paraphrases of each other?\n{options_}",
            targets="Let's see... {cot} So the answer is {answer}",
            inputs_prefix="Problem: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            in_template_mix=False),
    ],
    "sentiment140": [
        FewShotPattern(
            inputs="Write a tweet that is {answer} ({options_}).",
            targets="--- {text}",
            inputs_prefix="Question:\n",
            targets_prefix="Answer:\n",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{answer} ({options_})",
            targets="{text}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n"),
        FewShotPattern(
            inputs="Write a {answer} tweet.",
            targets="{text}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Generate a tweet. {answer}",
            targets="{text}",
            inputs_prefix="question: ",
            targets_prefix="answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{options_}. Generate a tweet that has the following sentiment: {answer}",
            targets="{text}",
            inputs_prefix="input question: ",
            targets_prefix="Generated tweet: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Write a {answer} tweet. Possible types: {options_}",
            targets="{text}",
            inputs_prefix="",
            targets_prefix="A: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Write a random tweet?",
            targets="{text}",
            inputs_prefix="input question: ",
            targets_prefix="A random tweet: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{options_}\nWrite a tweet that is {answer}.",
            targets="{text}",
            inputs_prefix="IN: ",
            targets_prefix="OUT: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{text}\n{options_}",
            targets="{answer}",
            inputs_prefix="Question:\n",
            targets_prefix="Answer:\n",
            x_y_delimiter="\n****\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Tweet: {text}\n{options_}",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="Sentiment: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n",
            in_template_mix=False),
    ],
    "synth_cot_sentiment140": [
        FewShotPattern(
            inputs="{text} ({options_}).",
            targets="{cot}",
            inputs_prefix="Question:\n",
            targets_prefix="Answer:\n",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{text} ({options_})",
            targets="My thought: {cot}. Answer: {answer}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n"),
        FewShotPattern(
            inputs="Sentiment of {text}? {options_}",
            targets="{cot}\nAnswer: {answer}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{text} Sentiment of the text before? {options_}",
            targets="My thought: {cot}. Answer: {answer}.",
            inputs_prefix="question: ",
            targets_prefix="answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Text: {text}\n{options_}",
            targets="{cot}\nSo the answer is: {answer}",
            inputs_prefix="[Q]: ",
            targets_prefix="[A]: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{text}\n{options_}",
            targets="Step-by-step reasoning: {cot}\nAnswer: {answer}",
            inputs_prefix="input: ",
            targets_prefix="output: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Think out loud: What is the sentiment?\nTweet:{text}\n{options_}",
            targets="{cot} The answer is {answer}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Tweet: {text}\nEXPLAIN the sentiment of this tweet.\n{options_}",
            targets="Answer: {answer}. Explanation: {cot}",
            inputs_prefix="input question: ",
            targets_prefix="output answer: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{text}\nIs this text positive or negative?\n{options_}\nWell, I think:",
            targets="{cot}\nSo the answer is: {answer}",
            inputs_prefix="Problem: ",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{text}\nAnalyze it:",
            targets="{cot}",
            inputs_prefix="Problem: ",
            targets_prefix="Short analysis: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
    ],
    "story_cloze": [
        FewShotPattern(
            inputs="Write a story that ends with this sentence.\n\n{answer}"
            "\n{options_}",
            targets="{context} {answer}",
            inputs_prefix="question: ",
            targets_prefix="answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{answer}\n{options_}",
            targets="{context} {answer}",
            inputs_prefix="Problem: ",
            targets_prefix="A: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{context} {options_}",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{context} {options_}",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Write a story that ends with: {answer}\n{options_}",
            targets="{context} {answer}",
            inputs_prefix="Problem: ",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n"),
        FewShotPattern(
            inputs="{options_}\nWrite a story that ends with: {answer}",
            targets="{context} {answer}",
            inputs_prefix="question: ",
            targets_prefix="answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Continue the story.\n\n{context}\n{options_}",
            targets="{answer}",
            inputs_prefix="Question:\n",
            targets_prefix="Story:\n",
            x_y_delimiter="\n",
            example_separator="\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{context}\n{options_}",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="Next sentence: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{context}\n{options_}",
            targets="{answer}",
            inputs_prefix="Input: ",
            targets_prefix="Completion: ",
            x_y_delimiter="\n-->\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{context}\n\nNow do a next sentence writing.\n{options_}",
            targets="{answer}",
            inputs_prefix="Problem: ",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n",
            in_template_mix=False),
    ],
    "copa": [
        FewShotPattern(
            inputs="Write a sentence.",
            targets="A: {premise}",
            inputs_prefix="Q: ",
            targets_prefix="",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Write two sentences. {options_}",
            targets="{answer} {premise}",
            inputs_prefix="question: ",
            targets_prefix="answer: ",
            x_y_delimiter="\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{premise} What is the {question}? {options_}",
            targets="{answer}",
            inputs_prefix="Question: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Here is a premise:{premise}\n\nWhat is the {question}?"
            "\n{options_}",
            targets="{answer}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{premise}\nWhat is the {question}?\n{options_}",
            targets="{answer}",
            inputs_prefix="[Q]: ",
            targets_prefix="[A]: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Write a sentence.",
            targets="{premise}",
            inputs_prefix="input question: ",
            targets_prefix="random sentence: ",
            x_y_delimiter="\n++++++++++\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Answer the following question about this sentence:\n\n{premise}\n\nWhat is the {question}?\n\n{options_}",
            targets="**{answer}**",
            inputs_prefix="",
            targets_prefix="Ans: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{premise}\n\n{question}: \n\n{options_}",
            targets="{answer}",
            inputs_prefix="[Q]: ",
            targets_prefix="[A]: ",
            x_y_delimiter="\n",
            example_separator="\n"),
        FewShotPattern(
            inputs="{question}?\n\n{premise}\n\n{options_}",
            targets="{answer}",
            inputs_prefix="[Q]: ",
            targets_prefix="[A]: ",
            x_y_delimiter="\n****\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{premise}\n\nWhat is a plausible {question}?\n\n{options_}",
            targets="{answer}",
            inputs_prefix="QUES:\n",
            targets_prefix="ANS:\n",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
    ],
    "winogrande": [
        FewShotPattern(
            inputs="How does the sentence end?\n\n{context}\n{options_}",
            targets="{answer}",
            inputs_prefix="Problem: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{context} {options_}",
            targets="{answer}",
            inputs_prefix="sentence: ",
            targets_prefix="complete: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{context} {options_}",
            targets="Continue the sentence -- {answer}",
            inputs_prefix="Question:\n",
            targets_prefix="Answer:\n",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{context}\n{options_}",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Continue writing.\n\n{context}\n{options_}",
            targets="{answer}",
            inputs_prefix="Input: ",
            targets_prefix="Continued: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{context}\n{options_}",
            targets="{answer}",
            inputs_prefix="Story needs to be completed:\n",
            targets_prefix="My choice:\n",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Continue the following story.\n\n{context}\n{options_}",
            targets="{answer}",
            inputs_prefix="Problem: ",
            targets_prefix="Here's how I want to continue it: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{context}\n{options_}",
            targets="{answer}",
            inputs_prefix="Problem: ",
            targets_prefix="My choice: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="How does the sentence end?\n\n{context}\n{options_}",
            targets="{answer}",
            inputs_prefix="[Q]: ",
            targets_prefix="[A]: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Continue writing.\n\n{context}\n\n{options_}",
            targets="{answer}",
            inputs_prefix="QUES: ",
            targets_prefix="ANS: ",
            x_y_delimiter="\n+++++++++\n",
            example_separator="\n\n\n"),
    ],
    "synth_cot_winogrande": [
        FewShotPattern(
            inputs="How does the sentence end?\n\n{context}\n{options_}",
            targets="{cot} The answer is {answer}.",
            inputs_prefix="Problem: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{context} {options_}",
            targets="{cot} The answer is {answer}.",
            inputs_prefix="sentence: ",
            targets_prefix="reasoning: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{context} {options_}",
            targets="{cot} The answer is {answer}.",
            inputs_prefix="Question:\n",
            targets_prefix="Answer:\n",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{context}\n{options_}",
            targets="{cot} The answer is {answer}.",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{context}\n\n{options_}",
            targets="{cot} The answer is {answer}.",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{context}\n{options_}\nLet's think step-by-step.",
            targets="{cot} The answer is {answer}.",
            inputs_prefix="[Q]: ",
            targets_prefix="[Step-by-step]: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{context}\n{options_}",
            targets="{cot}\nThe answer is {answer}.",
            inputs_prefix="",
            targets_prefix="So, let's think step-by-step:",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Continue writing the following text.\n\n{context}\n\n{options_}\nWell...",
            targets="{cot} So the answer is {answer}.",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="How does the sentence end?{context}\n\n{options_}",
            targets="{cot} The answer is {answer}.",
            inputs_prefix="Question: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n****\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Write the next sentence.\n{context}\n{options_}",
            targets="Step-by-step reasoning process: {cot}\nThe answer is {answer}.",
            inputs_prefix="input question: ",
            targets_prefix="output answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            in_template_mix=False),
    ],
    "yelp_polarity_reviews": [
        FewShotPattern(
            inputs="{text} {options_}",
            targets="{answer}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Write a {answer} ({options_}) yelp review.",
            targets="{text}",
            inputs_prefix="Problem: ",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{answer}",
            targets="{text}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="What would be an example of an {answer} ({options_}) "
            "review?",
            targets="{text}",
            inputs_prefix="Problem: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{options_} What would be an example of an {answer} review?",
            targets="An example of an {answer} review: {text}",
            inputs_prefix="Input: ",
            targets_prefix="",
            x_y_delimiter="\n\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{options_}.\nGenerate a {answer} review for a place",
            targets="{text}",
            inputs_prefix="Input: ",
            targets_prefix="Output: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Write a {answer} yelp review ({options_}).",
            targets="{text}",
            inputs_prefix="input: ",
            targets_prefix="output: ",
            x_y_delimiter="\n",
            example_separator="\n"),
        FewShotPattern(
            inputs="{text}\n{options_}",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Sentiment analysis: {text}\n\n{options_}",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n+++++++++\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{text}\nHow would this review be described?\n{options_}",
            targets="{answer}",
            inputs_prefix="Problem: ",
            targets_prefix="A: ",
            x_y_delimiter="\n++++++++++\n",
            example_separator="\n\n\n",
            in_template_mix=False),
    ],
    "arc": [
        FewShotPattern(
            inputs="{question} {options_}",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Write a question you would see in a school textbook."
            " {options_}",
            targets="{question}",
            inputs_prefix="question: ",
            targets_prefix="answer: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="An example of a grad-school level question?",
            targets="{question}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="I just took a test in school today. What question was I "
            "asked?",
            targets="{question}",
            inputs_prefix="",
            targets_prefix="Question I was asked: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Random question?",
            targets="{question}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Generate a question",
            targets="{question}",
            inputs_prefix="",
            targets_prefix="Question generated: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Write a question you would see in a school textbook.",
            targets="{question}",
            inputs_prefix="Problem: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{question}\n\n{options_}",
            targets="{answer}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Answer the question\n\n{question}\n{options_}",
            targets="{answer}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Choose your answer.\n\n{question}\n\n{options_}",
            targets="{answer}",
            inputs_prefix="Question: ",
            targets_prefix="My Answer: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
    ],
    "anli": [
        FewShotPattern(
            inputs="Generate a context and a hypothesis.",
            targets="Context: {context}\n\nHypothesis: {hypothesis}",
            inputs_prefix="",
            targets_prefix="Answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{context}\n\nBased on the paragraph above can we conclude "
            "that \"{hypothesis}\"? {options_}",
            targets="{answer}",
            inputs_prefix="Problem: ",
            targets_prefix="A: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{hypothesis}\n{context}\n{options_}",
            targets="{answer}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n"),
        FewShotPattern(
            inputs="{context}\n{hypothesis} {options_}",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Generate a context and a hypothesis.",
            targets="Context: {context}\n\nHypothesis: {hypothesis}",
            inputs_prefix="Q: ",
            targets_prefix="Generated: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{hypothesis}\nContext: {context}\n{options_}",
            targets="{answer}",
            inputs_prefix="input hypothesis: ",
            targets_prefix="true or false: ",
            x_y_delimiter="\n",
            example_separator="\n"),
        FewShotPattern(
            inputs="Context:\n{context}\nHypothesis: {hypothesis} {options_}",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n"),
        FewShotPattern(
            inputs="{options_}\n\n{context}\n\nSentence: {hypothesis}",
            targets="{answer}",
            inputs_prefix="Input: ",
            targets_prefix="Output: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Read the following paragraph and determine if the hypothesis is true:\n\n{context}\n\n{options_}\nHypothesis: {hypothesis}",
            targets="{answer}",
            inputs_prefix="Problem:\n",
            targets_prefix="Answer:\n",
            x_y_delimiter="\n****\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{context} {hypothesis} {options_}",
            targets="{answer}",
            inputs_prefix="[Q]: ",
            targets_prefix="[A]: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
    ],
    "synth_cot_anli": [
        FewShotPattern(
            inputs="{premise}\n{hypothesis}\n",
            targets="{cot} The answer is {answer}",
            inputs_prefix="",
            targets_prefix="Answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{premise}\n\nBased on the paragraph above can we conclude "
            "that \"{hypothesis}\"? {options_}",
            targets="{cot} The answer is {answer}",
            inputs_prefix="Problem: ",
            targets_prefix="A: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{hypothesis}\n{premise}\n{options_}",
            targets="{cot}\n...the answer is {answer}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{premise}\n{hypothesis} {options_}",
            targets="{cot} The answer is {answer}",
            inputs_prefix="",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Generate a premise and a hypothesis with explanation",
            targets="Context: {premise}\nHypothesis: {hypothesis}\n{options_}\nExplanation: {cot} The answer is {answer}",
            inputs_prefix="",
            targets_prefix="Generated: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Context:\n\n{premise}\n\nHypothesis: {hypothesis}\n{options_}",
            targets="Let me think. {cot} The answer is {answer}",
            inputs_prefix="[Q]: ",
            targets_prefix="[A]: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Read the text and determine if the sentence is true:\n{premise} Sentence: {hypothesis} {options_}",
            targets="{cot} The answer is {answer}",
            inputs_prefix="Input: ",
            targets_prefix="Output: ",
            x_y_delimiter="\n++++++++++\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{premise} {options_} Hypothesis: {hypothesis}",
            targets="{cot} The answer is {answer}",
            inputs_prefix="Problem: ",
            targets_prefix="A: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{premise}\nDoes this next sentence follow, given the preceding text?\n{hypothesis}\n\n{options_}",
            targets="{cot} The answer is {answer}",
            inputs_prefix="[Q]: ",
            targets_prefix="[A]: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{premise}\n\n{hypothesis}\n\n{options_}",
            targets="{cot} The answer is {answer}",
            inputs_prefix="",
            targets_prefix="Answer with step-by-step reasoning: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n"),
    ],
    "coqa": [
        FewShotPattern(
            inputs="{text}\n{numbered_questions}",
            targets="{numbered_answers}",
            inputs_prefix="Problem: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{text}\n\nAnswer the following questions:"
            "\n{numbered_questions}",
            targets="{numbered_answers}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n"),
        FewShotPattern(
            inputs="Read the text and answer the questions."
            "\n{text}\n{numbered_questions}",
            targets="Numbered answers:\n{numbered_answers}",
            inputs_prefix="Question:\n",
            targets_prefix="",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Answer the questions at the end based on the text."
            "\n\n{text}\n\n{numbered_questions}\n\nNumbered answers:",
            targets="{numbered_answers}",
            inputs_prefix="",
            targets_prefix="A: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{text} {numbered_questions} Provide a numbered list of answers.",
            targets="{numbered_answers}",
            inputs_prefix="Question: ",
            targets_prefix="A numbered of answers: ",
            x_y_delimiter="\n****\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{text}\n\n{numbered_answers}\n\nNumbered questions:",
            targets="{numbered_questions}",
            inputs_prefix="Q: ",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Make use of the article to answer the questions. {text} {numbered_questions}",
            targets="{numbered_answers}",
            inputs_prefix="input: ",
            targets_prefix="numbered_answers: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{text} {numbered_questions} Return numbered answers in your output.",
            targets="{numbered_answers}",
            inputs_prefix="input: ",
            targets_prefix="output: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{text}\n{numbered_questions}",
            targets="{numbered_answers}",
            inputs_prefix="question: ",
            targets_prefix="answer: ",
            x_y_delimiter="\n****\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{text} What are the answers to this following set of questions: {numbered_questions}",
            targets="{numbered_answers}",
            inputs_prefix="",
            targets_prefix="Answer: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
    ],
    "opinion_abstracts_rotten_tomatoes": [
        FewShotPattern(
            inputs="{numbered_reviews}",
            targets="{critic_consensus}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{critic_consensus}",
            targets="{numbered_reviews}",
            inputs_prefix="Critic consensus:",
            targets_prefix="Numbered reviews:",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{numbered_reviews}\nBased on these individual reviews, "
            "what is the critic consensus?",
            targets="Consensus: {critic_consensus}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{numbered_reviews}",
            targets="Consensus: {critic_consensus}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n"),
        FewShotPattern(
            inputs="{critic_consensus}",
            targets="{numbered_reviews}",
            inputs_prefix="question: ",
            targets_prefix="numbered reviews: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Write an ordered list of reviews about \"{movie}\".",
            targets="{numbered_reviews}",
            inputs_prefix="IN: ",
            targets_prefix="OUT: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Write a very short review for movie \"{movie}\".",
            targets="{critic_consensus}",
            inputs_prefix="input: ",
            targets_prefix="output: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Here are some reviews for a movie: {numbered_reviews}",
            targets="{critic_consensus}",
            inputs_prefix="",
            targets_prefix="Here is the consensus of critics:",
            x_y_delimiter="\n\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{numbered_reviews}\nWhat is the consensus?",
            targets="{critic_consensus}",
            inputs_prefix="Problem: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n****\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{numbered_reviews}\nWhat is the critic consensus?",
            targets="{critic_consensus}",
            inputs_prefix="",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
    ],
    "opinion_abstracts_idebate": [
        FewShotPattern(
            inputs="{argument_sentences}",
            targets="{claim}",
            inputs_prefix="Argument:",
            targets_prefix="Claim: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{argument_sentences}\nWhat claim can be made from these "
            "sentences?",
            targets="{claim}",
            inputs_prefix="Sentences: ",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{debate_name}\nWhat argument could one make about this "
            "debate topic?",
            targets="Claim: [{claim}]",
            inputs_prefix="",
            targets_prefix="A: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{debate_name}",
            targets="{claim}",
            inputs_prefix="",
            targets_prefix="Claim: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Claim: {claim}\nWhat evidence supports this claim?",
            targets="{argument_sentences}",
            inputs_prefix="PROBLEM: ",
            targets_prefix="EVIDENCE: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{claim} Come up with some evidence to support this.",
            targets="{argument_sentences}",
            inputs_prefix="Problem: ",
            targets_prefix="Evidence: ",
            x_y_delimiter="\n++++++++++\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{argument_sentences}",
            targets="{debate_name}",
            inputs_prefix="Arguments: ",
            targets_prefix="Debate: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="What debate topic are the following sentences about? {argument_sentences}",
            targets="{debate_name}",
            inputs_prefix="Question: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n****\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Summarize the argument implied by these sentences. {argument_sentences}",
            targets="{claim}",
            inputs_prefix="Question:\n",
            targets_prefix="Answer:\n",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="What claim can be made from the following pieces of evidence?\n{argument_sentences}",
            targets="{claim}",
            inputs_prefix="Question: ",
            targets_prefix="Claim: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
    ],
    "common_gen": [
        FewShotPattern(
            inputs="Concepts: {concepts}",
            targets="{target}",
            inputs_prefix="",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n"),
        FewShotPattern(
            inputs="Keywords: {concepts}\nWhat is a sentence that includes all "
            "these keywords?",
            targets="{target}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Here are some concepts: {concepts}\n\nWhat is a sentence "
            "about these concepts?",
            targets="Sentence is below.\n{target}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Produce a sentence which mentions all of these concepts: "
            "{concepts}",
            targets="{target}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Generate a sentence, and then tell me the concepts included in that sentence.",
            targets="Sentence: {target}\nConcepts: {concepts_newline}",
            inputs_prefix="input question: ",
            targets_prefix="output answer: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Identify the most salient words:\n\n{target}",
            targets="{concepts_newline}",
            inputs_prefix="Input: ",
            targets_prefix="Output: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="What are the most important words?\n{target}",
            targets="{concepts}",
            inputs_prefix="Problem: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="What are the keywords in the following sentence? {target}",
            targets="{concepts}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Generate a sentence that includes all the following words: {concepts}",
            targets="{target}",
            inputs_prefix="[Q]: ",
            targets_prefix="[A]: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{concepts}",
            targets="{target}",
            inputs_prefix="CONCEPTS: ",
            targets_prefix="GEN: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n"),
    ],
    "synth_cot_common_gen": [
        FewShotPattern(
            inputs="Concepts: {concepts}",
            targets="{cot} The answer is {target}",
            inputs_prefix="",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n"),
        FewShotPattern(
            inputs="Keywords: {concepts}\nWhat is a sentence that includes all "
            "these keywords? Let add some explanation.",
            targets="{cot} The answer is {target}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Here are some concepts: {concepts}\n\nWhat is a sentence "
            "about these concepts?",
            targets="Chain-of-thought: {cot} The answer is {target}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Produce a sentence which mentions all of these concepts: "
            "{concepts}",
            targets="{cot} The answer is {target}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{concepts_newline}",
            targets="{cot} The answer is {target}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n++++++++++\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="A sentence generation problem: {concepts}",
            targets="{cot} The answer is {target}",
            inputs_prefix="Input: ",
            targets_prefix="Output: ",
            x_y_delimiter="\n",
            example_separator="\n"),
        FewShotPattern(
            inputs="Concepts: {concepts}. Now, generate something from these concepts.",
            targets="{cot}\nThe answer is {target}",
            inputs_prefix="input question: ",
            targets_prefix="generation process: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Here are some concepts: {concepts} Write a sentence about these concepts.",
            targets="{cot}\nThe answer is {target}",
            inputs_prefix="",
            targets_prefix="Let's think: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Keywords: {concepts} What is a sentence that includes all these keywords? Let see...",
            targets="{cot} The answer is {target}",
            inputs_prefix="Problem: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Generate a sentence using concepts: {concepts_newline}. Think step-by-step:",
            targets="{cot}\nThe answer is:\n\n{target}",
            inputs_prefix="",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n",
            in_template_mix=False),
    ],
    "dart": [
        FewShotPattern(
            inputs="Triple: {tripleset}",
            targets="{target}",
            inputs_prefix="question: ",
            targets_prefix="answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Data: {tripleset}\n\nWhat would a sentence about this data "
            "be like?",
            targets="{target}",
            inputs_prefix="Problem: ",
            targets_prefix="A: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Generate an approximately fifteen-word sentence that "
            "describes all this data: {tripleset}",
            targets="Answer: {target}",
            inputs_prefix="question: ",
            targets_prefix="",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="This is some data: {tripleset}.\nGenerate a detailed "
            "description of this data.",
            targets="{target}",
            inputs_prefix="Question:\n",
            targets_prefix="Answer:\n",
            x_y_delimiter="\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Create a set of triples that describes the content in the following sentence. {target}",
            targets="{tripleset_newline}",
            inputs_prefix="Question: ",
            targets_prefix="A set of triples: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="What concepts are described in the following sentence? \"{target}\" Return the answer as pairs of triples.",
            targets="{tripleset_newline}",
            inputs_prefix="[Question]: ",
            targets_prefix="[Tripleset]: ",
            x_y_delimiter="\n",
            example_separator="\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{tripleset}",
            targets="{target}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="*{tripleset}*",
            targets="{target}",
            inputs_prefix="Problem: ",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n"),
        FewShotPattern(
            inputs="This is some data: {tripleset}. Generate a detailed description of this data.",
            targets="{target}",
            inputs_prefix="Input: ",
            targets_prefix="Output: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Generate an approximately fifteen-word sentence that describes all of {tripleset}",
            targets="{target}",
            inputs_prefix="[Q]: ",
            targets_prefix="[A]: ",
            x_y_delimiter="\n****\n",
            example_separator="\n\n\n"),
    ],
    "e2e_nlg": [
        FewShotPattern(
            inputs="Attributes: {meaning_representation}. Produce a detailed "
            "sentence about this restaurant.",
            targets="[{target}]",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{meaning_representation}.",
            targets="{target}",
            inputs_prefix="Data: ",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Data: {meaning_representation}.",
            targets="** {target}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Here are some keywords about a restaurant:\n\n"
            "{meaning_representation}.",
            targets="{target}",
            inputs_prefix="",
            targets_prefix="A: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Sentence: {meaning_representation}\nPlz represent the content in this sentence in data form.",
            targets="{target}",
            inputs_prefix="[Q]: ",
            targets_prefix="[A]: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Some data about a restaurant: {meaning_representation}. Write a sentence that includes the above data.",
            targets="{target}",
            inputs_prefix="Question:\n",
            targets_prefix="Answer:\n",
            x_y_delimiter="\n",
            example_separator="\n"),
        FewShotPattern(
            inputs="{meaning_representation}.",
            targets="{target}",
            inputs_prefix="",
            targets_prefix="Answer: ",
            x_y_delimiter="\n****\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Data: {meaning_representation}.",
            targets="{target}",
            inputs_prefix="",
            targets_prefix="A sentence that describes this data: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Data: {meaning_representation}. Can you generate a sentence?",
            targets="{target}",
            inputs_prefix="Problem: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Generate Problem with Attributes: {meaning_representation}",
            targets="{target}",
            inputs_prefix="",
            targets_prefix="Generation: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
    ],
    "web_nlg_en": [
        FewShotPattern(
            inputs="{input_string}",
            targets="{target}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{input_string}",
            targets="{target}",
            inputs_prefix="Data: ",
            targets_prefix="",
            x_y_delimiter="\n\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Sentence: {target}\n\nWhat data can be extracted from this "
            "sentence?",
            targets="{input_string}",
            inputs_prefix="",
            targets_prefix="A: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Sentence: {target}\n\nQuestion: What structured data could "
            "we extract from this sentence?",
            targets="{input_string}",
            inputs_prefix="",
            targets_prefix="A: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Sentence: {target}",
            targets="{input_string}",
            inputs_prefix="input question: ",
            targets_prefix="Structured data: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Sentence: {target} What data can be extracted from this sentence?",
            targets="{input_string}",
            inputs_prefix="",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Data: {input_string}\nSentence about the following data:",
            targets="{target}",
            inputs_prefix="",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="What is sentence that verbalizes this data: {input_string}",
            targets="{target}",
            inputs_prefix="question: ",
            targets_prefix="sentence: ",
            x_y_delimiter="\n***\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Generate an approximately fifteen-word sentence that describes all this data: {input_string}",
            targets="{target}",
            inputs_prefix="input: ",
            targets_prefix="sentence: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Produce a long descriptive sentence that uses all these words: {input_string}",
            targets="{target}",
            inputs_prefix="",
            targets_prefix="A: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n",
            in_template_mix=False),
    ],
    "wiki_lingua_english_en": [
        FewShotPattern(
            inputs="{source}",
            targets="{target}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Article: {source}\nQuestion: What is a summary of what "
            "this article is about?",
            targets="{target}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n"),
        FewShotPattern(
            inputs="{source}",
            targets="Summary: {target}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Summarize the following:\n{source}",
            targets="summary: {target}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Article: {source}",
            targets="{target}",
            inputs_prefix="Problem: ",
            targets_prefix="Summary: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Article: {source}",
            targets="{target}",
            inputs_prefix="INPUT ARTICLE: ",
            targets_prefix="SUMMARY: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Write an article based on this \"{target}\"",
            targets="{source}",
            inputs_prefix="",
            targets_prefix="article: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Write an article based on this summary: {target}",
            targets="{source}",
            inputs_prefix="Problem: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="In one sentence, describe what the following article is about: {source}",
            targets="{target}",
            inputs_prefix="",
            targets_prefix="Summary: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{source}",
            targets="{target}",
            inputs_prefix="",
            targets_prefix="One-sentence summary -- ",
            x_y_delimiter="\n++++++++++\n",
            example_separator="\n\n\n"),
    ],
    "multirc": [
        FewShotPattern(
            inputs="{paragraph}\n\nQuestion: \"{question}\"\n\n"
            "{options_}\n\nAnswer: \"{response}\"",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="Response: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{paragraph}",
            targets="{question}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{paragraph}\nWhat question would one ask from this "
            "paragraph?",
            targets="{question}",
            inputs_prefix="",
            targets_prefix="Answer: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{paragraph}\n\nQuestion: \"{question}\"\n\n"
            "Response: \"{response}\"\n{options_}",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="A: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{paragraph}\nWhat question would one ask from this paragraph?",
            targets="{question}",
            inputs_prefix="",
            targets_prefix="QUESTION: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{paragraph}\nDo you have any questions?",
            targets="{question}",
            inputs_prefix="QUES: ",
            targets_prefix="MY QUESTION: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{paragraph}\n\nQuestion: \"{question}\"\n\nAnswer: \"{response}\"\n\n{options_}",
            targets="{answer}",
            inputs_prefix="Problem: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{paragraph} After reading the above, is \"{response}\" the correct answer to the question \"{question}\"? {options_}",
            targets="{answer}",
            inputs_prefix="Question:\n",
            targets_prefix="Answer:\n",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{paragraph}\n\"{question}\" is \"{response}\"?\n\n{options_}",
            targets="{answer}",
            inputs_prefix="Input: ",
            targets_prefix="Output: ",
            x_y_delimiter="\n",
            example_separator="\n"),
        FewShotPattern(
            inputs="{paragraph}\n\nDoes the response \"{response}\" correctly answer the question \"{question}\"?\n\n{options_}",
            targets="{answer}",
            inputs_prefix="Input: ",
            targets_prefix="Output: ",
            x_y_delimiter="\n***\n",
            example_separator="\n\n\n"),
    ],
    "cb": [
        FewShotPattern(
            inputs="{hypothesis}\n\n{premise}\n{options_}",
            targets="{answer}",
            inputs_prefix="question: ",
            targets_prefix="answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Generate a context and a hypothesis.",
            targets="Context: {premise}\n\nHypothesis: {hypothesis}",
            inputs_prefix="",
            targets_prefix="Answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{premise}\n\nBased on the paragraph above can we conclude "
            "that \"{hypothesis}\"? {options_}",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{premise}\n{hypothesis}\n{options_}",
            targets="{answer}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Generate a context and a hypothesis.",
            targets="Context: {premise}\n\nHypothesis: {hypothesis}",
            inputs_prefix="INSTRUCTION: ",
            targets_prefix="GENERATED: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{hypothesis}\n{options_}\nSo, is the hypothesis above true, given the following?\n{premise}",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="Answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Context:\n\n{premise}\n\nHypothesis: {hypothesis}\n{options_}",
            targets="{answer}",
            inputs_prefix="QUESTION: ",
            targets_prefix="ANS: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Read the text and determine if the sentence is true:\n\n{premise}\n\nSentence: {hypothesis}\n{options_}",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="A: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Read the following paragraph and determine if the hypothesis is true: {premise}\nHypothesis: {hypothesis}\n{options_}",
            targets="{answer}",
            inputs_prefix="Input: ",
            targets_prefix="Output: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{premise} Can we infer the following? {hypothesis} {options_}",
            targets="{answer}",
            inputs_prefix="Problem: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n"),
    ],
    "cola": [
        FewShotPattern(
            inputs="{answer}",
            targets="{sentence}",
            inputs_prefix="Question: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n",
            example_separator="\n"),
        FewShotPattern(
            inputs="Produce a sentence that would be considered grammatically "
            "{answer} ({options_})",
            targets="{sentence}",
            inputs_prefix="Problem: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Sentence: \"{sentence}\"\nWould a linguist rate this "
            "sentence to be acceptable linguistically? {options_}",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{sentence}\n{options_}",
            targets="Linguistic integrity: {answer}",
            inputs_prefix="Sentence: ",
            targets_prefix="",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Produce a brief English sentence that would be considered grammatically as category: {answer}\nAll categories: {options_}",
            targets="{sentence}",
            inputs_prefix="Instruction: ",
            targets_prefix="A brief sentence: ",
            x_y_delimiter="\n",
            example_separator="\n"),
        FewShotPattern(
            inputs="{answer} ({options_})",
            targets="{sentence}",
            inputs_prefix="QUES: ",
            targets_prefix="ANS: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Valid English sentence (grammatically) or not: {sentence}\n{options_}",
            targets="{answer}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Is the next sentence syntactically and semantically acceptable?\n\n{sentence}\n{options_}",
            targets="{answer}",
            inputs_prefix="instruction: ",
            targets_prefix="response: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Would the following sentence, by the strictest standards, be considered correct? {sentence} {options_}",
            targets="{answer}",
            inputs_prefix="[Q]: ",
            targets_prefix="[A]: ",
            x_y_delimiter="\n++++++++++\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Is the following sentence linguistically acceptable?\n{sentence}\n{options_}",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n\n",
            example_separator="\n\n",
            in_template_mix=False),
    ],
    "sst2": [
        FewShotPattern(
            inputs="Positive or negative opinion of the movie?\n\n{sentence}"
            "\n{options_}",
            targets="Answer: {answer}",
            inputs_prefix="Question: ",
            targets_prefix="--",
            x_y_delimiter="\n",
            example_separator="\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Write a {answer} ({options_}) movie review.",
            targets="{sentence}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Generate a short movie review that has {answer} sentiment.",
            targets="{sentence}",
            inputs_prefix="problem: ",
            targets_prefix="generated: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Review:\n{sentence}\nIs this review negative or positive?"
            "\n{options_}",
            targets="{answer}",
            inputs_prefix="question: ",
            targets_prefix="answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Generate a short movie review that has \"{answer}\" sentiment\n{options_}.",
            targets="{sentence}",
            inputs_prefix="input: ",
            targets_prefix="output: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Write a movie review.",
            targets="{sentence}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n"),
        FewShotPattern(
            inputs="How's the review for the movie?\n\n{sentence}\n{options_}",
            targets="{answer}",
            inputs_prefix="QUES: ",
            targets_prefix="ANS: ",
            x_y_delimiter="\n++++++++++\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{options_} Would the following phrase be considered positive or negative?\n\n{sentence}",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Is the sentiment of the following sentence positive or negative? {sentence}\n{options_}",
            targets="{answer}",
            inputs_prefix="Question:\n",
            targets_prefix="Answer:\n",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Sentence from a movie review: {sentence}\n{options_}",
            targets="{answer}",
            inputs_prefix="question: ",
            targets_prefix="answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            in_template_mix=False),
    ],
    "mnli": [
        FewShotPattern(
            inputs="Premise: {premise}\n\nHypothesis: {hypothesis}\n{options_}",
            targets="Answer: {answer}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{premise}\n{hypothesis}\n{options_}",
            targets="{answer}",
            inputs_prefix="Question: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Here is a premise:\n{premise}\n\nHere is a hypothesis:"
            "\n{hypothesis}\n\nHere are the options: {options_}\n",
            targets="{answer}",
            inputs_prefix="Problem: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Sentence 1: {premise}\n\nSentence 2: {hypothesis}\n"
            "{options_}\nIs this second sentence entailed by the first?",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="Answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{premise}\n\nDoes it follow that \"{hypothesis}\"?\n{options_}",
            targets="{answer}",
            inputs_prefix="Question:\n",
            targets_prefix="Answer:\n",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="If {premise}, can we say that \"{hypothesis}\"?\n{options_}",
            targets="{answer}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Premise:\n\"{premise}\"\nHypothesis: {hypothesis}\n{options_}",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{premise} If this premise is true, does that tell us:\"{hypothesis}\"?\n\n{options_}",
            targets="{answer}",
            inputs_prefix="input question: ",
            targets_prefix="output answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Based on the premise \"{premise}\", can we conclude that \"{hypothesis}\"? {options_}",
            targets="{answer}",
            inputs_prefix="Question:\n",
            targets_prefix="Answer:\n",
            x_y_delimiter="\n****\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Sentence A: {premise}\n\nSentence B: {hypothesis}\n\nIf sentence A is true, how about sentence B?\n{options_}",
            targets="{answer}",
            inputs_prefix="Question: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
    ],
    "qnli": [
        FewShotPattern(
            inputs="Generate a question with a factual answer.",
            targets="Generated: {question}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Does \"{sentence}\" answer the question \"{question}\" "
            "{options_}",
            targets="{answer}",
            inputs_prefix="Problem: ",
            targets_prefix="A: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Does the sentence \"{sentence}\" provide a valid answer to "
            "the question \"{question}\" {options_}",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="A: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Choose your answer: Is \"{sentence}\" a good answer to the "
            "question \"{question}\" {options_}",
            targets="{answer}",
            inputs_prefix="Question:\n",
            targets_prefix="Answer:\n",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Generate a question with a factual answer?",
            targets="{question}",
            inputs_prefix="QUESTION: ",
            targets_prefix="ANS: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Q: {question}\nA: {sentence}\nDoes the answer answer the question? {options_}",
            targets="{answer}",
            inputs_prefix="QUES: ",
            targets_prefix="ANS: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Does \"{sentence}\" contain the correct answer to \"{question}\"\n{options_}",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="A: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{options_}\n{sentence}\n{question}",
            targets="{answer}",
            inputs_prefix="question: ",
            targets_prefix="answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Sentence: {sentence} Question: {question} {options_}",
            targets="{answer}",
            inputs_prefix="[Q]: ",
            targets_prefix="[A]: ",
            x_y_delimiter="\n++++++++++\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Does \"{sentence}\" provide a valid answer to \"{question}\"?\n{options_}",
            targets="{answer}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
    ],
    "wnli": [
        FewShotPattern(
            inputs="Sentence 1: \"{sentence1}\"\nSentence 2: \"{sentence2}\"\n"
            "Is sentence 2 true, based on sentence 1? {options_}",
            targets="ANS:\n{answer}",
            inputs_prefix="Question:\n",
            targets_prefix="",
            x_y_delimiter="\n\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="If \"{sentence1}\", can we conclude that \"{sentence2}\"?"
            " {options_}",
            targets="{answer}",
            inputs_prefix="Question: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="If \"{sentence1}\", does it follow that \"{sentence2}\"? "
            "{options_}",
            targets="{answer}",
            inputs_prefix="Problem: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="If \"{sentence1}\", is \"{sentence2}\" "
            "correct?\n{options_}",
            targets="{answer}",
            inputs_prefix="Problem: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="SA: \"{sentence1}\"\n\nSB: \"{sentence2}\"\n\nIs SB true, based on SA?\n{options_}",
            targets="{answer}",
            inputs_prefix="Question:\n",
            targets_prefix="Answer:\n",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Is \"{sentence2}\" true if \"{sentence1}\"?\n\n{options_}",
            targets="{answer}",
            inputs_prefix="Problem: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Can we say \"{sentence2}\" if \"{sentence1}\"?\n\n{options_}",
            targets="{answer}",
            inputs_prefix="Input: ",
            targets_prefix="Output: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Is it true that \"{sentence2}\" if \"{sentence1}\" is true? {options_}",
            targets="{answer}",
            inputs_prefix="input question: ",
            targets_prefix="output answer: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Sentence: \"{sentence2}\"; Another sentence: \"{sentence1}\"?\n\n{options_}",
            targets="{answer}",
            inputs_prefix="Problem: ",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="\"{sentence1}\" is true.\nSo, is \"{sentence2}\" true as well?\n\n{options_}",
            targets="{answer}",
            inputs_prefix="question: ",
            targets_prefix="prediction: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
    ],
    "snli": [
        FewShotPattern(
            inputs="Is the premise \"{premise}\" true if \"{hypothesis}\"?"
            "\n{options_}",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n"),
        FewShotPattern(
            inputs="Write a brief sentence.",
            targets="{hypothesis}",
            inputs_prefix="Problem: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n",
            example_separator="\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="\"{premise}\" Does this mean that \"{hypothesis}\"? "
            "{options_}",
            targets="{answer}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{premise}\n{hypothesis}\n{options_}",
            targets="{answer}",
            inputs_prefix="Premise & Hypothesis & Options: ",
            targets_prefix="Is the hypothesis true or not: ",
            x_y_delimiter="\n",
            example_separator="\n\n\n\n"),
        FewShotPattern(
            inputs="Write a brief sentence.",
            targets="{hypothesis}",
            inputs_prefix="question: ",
            targets_prefix="answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Is the premise \"{premise}\" true if \"{hypothesis}\"?\n{options_}",
            targets="{answer}",
            inputs_prefix="Premise & hypothesis: ",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{premise}\n\n{hypothesis}\n{options_}",
            targets="{answer}",
            inputs_prefix="Premise & hypothesis.\n",
            targets_prefix="true or not.\n",
            x_y_delimiter="\n++++++++++\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Premise: {premise}\n\nHypothesis: {hypothesis}\n{options_}",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Premise: {premise}\n\nHypothesis: {hypothesis}\nIs the hypothesis true?\n{options_}",
            targets="{answer}",
            inputs_prefix="Question:\n",
            targets_prefix="Answer:\n",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Based on the sentence \"{premise}\", is the sentence \"{hypothesis}\" true?\n\n{options_}",
            targets="{answer}",
            inputs_prefix="Question: ",
            targets_prefix="Answer:\n",
            x_y_delimiter="\n",
            example_separator="\n\n"),
    ],
    "trec": [
        FewShotPattern(
            inputs="Please ask me a question.",
            targets="\nMe: {text}",
            inputs_prefix="You: ",
            targets_prefix="",
            x_y_delimiter="\n\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="What type of thing is the question \"{text}\" asking about?"
            "\n{options_}",
            targets="{answer}",
            inputs_prefix="Problem: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Is the question \"{text}\" asking about an entity, an "
            "abbreviation, a description, a human, a location, or a "
            "numeric entity?",
            targets="{answer}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{options_}\n\"{text}\"",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n"),
        FewShotPattern(
            inputs="Please ask me a question.",
            targets="{text}",
            inputs_prefix="Input: ",
            targets_prefix="Output: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{text}\n\nWould the answer to this question be an entity, an abbreviation, a description, a human, a location, or a numeric value?\n\n{options_}",
            targets="{answer}",
            inputs_prefix="input question: ",
            targets_prefix="your answer: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{text}\n\nWhat kind of thing would answer this question?\n\n{options_}",
            targets="{answer}",
            inputs_prefix="Input: ",
            targets_prefix="Output: ",
            x_y_delimiter="\n***\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="What is the question \"{text}\" asking about?\n\n{options_}",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{text}\n{options_}",
            targets="{answer}",
            inputs_prefix="Part1.\n",
            targets_prefix="Part2.\n",
            x_y_delimiter="\n****\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{options_} So, would the answer to \"{text}\" be an abbreviation, an entity, a human, a human, a description, or a numeric value?",
            targets="{answer}",
            inputs_prefix="QUES: ",
            targets_prefix="ANS: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
    ],
    "stsb": [
        FewShotPattern(
            inputs="{sentence1}\n{answer}\n{options_}",
            targets="{sentence2}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{sentence2}\n\nWhat is a sentence that would be (on a "
            "scale from 0 to 5) a {answer} out of 5 in terms of textual "
            "similarity to the above sentence? {options_}",
            targets="{sentence1}",
            inputs_prefix="Question:\n",
            targets_prefix="Answer:\n",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{sentence1}\n{sentence2}\n\nRate the textual similarity of "
            "these two sentences on a scale from 0 to 5, where 0 is "
            "\"no meaning overlap\" and 5 is \"means the same thing\". "
            "{options_}",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="Answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{sentence1}\n{sentence2}\n{options_}",
            targets="{answer}",
            inputs_prefix="Question:\n",
            targets_prefix="Answer:\n",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{sentence2}\n\nWhat is a sentence that would be (on a scale from 0 to 5) a {answer} out of 5 in terms of textual similarity to the above sentence? {options_}",
            targets="{sentence1}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{sentence1}\n\nGenerate a new sentence, on a scale from 0 to 5, a {answer} ({options_}) in textual similarity to the above sentence.",
            targets="{sentence2}",
            inputs_prefix="Question: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{sentence1}\n{sentence2}\n\n{options_}",
            targets="{answer}",
            inputs_prefix="[Q]: ",
            targets_prefix="[A]: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="On a scale from 0-5, where 0 is \"not similar\" and 5 is \"very similar\", how similar is the sentence \"{sentence1}\" to the sentence \"{sentence2}\"?\n\n{options_}",
            targets="{answer}",
            inputs_prefix="Input: ",
            targets_prefix="Output: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Rate the similarity of the following two sentences\n{sentence1}\n{sentence2}\n\n{options_}",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Do the following sentences say the same thing?\n{options_}\n{sentence1}\n{sentence2}",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n\n",
            example_separator="\n\n"),
    ],
    "hellaswag": [
        FewShotPattern(
            inputs="What happens next in this paragraph?\n\n{context}"
            "\n{options_}",
            targets="{answer}",
            inputs_prefix="Problem: ",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Problem: {context} {options_}",
            targets="Answer: {answer}",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{context}\n\n{options_}",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Complete the next sentence:\n\n{context}\n{options_}",
            targets="{answer}",
            inputs_prefix="question: ",
            targets_prefix="answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Write the next sentence for:\n{context}\n\n{options_}.",
            targets="{answer}",
            inputs_prefix="Problem: ",
            targets_prefix="Next sentence: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{context}\n\n{options_}",
            targets="{answer}",
            inputs_prefix="context: ",
            targets_prefix="next sentence for the context: ",
            x_y_delimiter="\n****\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="What happens next?\n\n{context}\n\n{options_}",
            targets="{answer}",
            inputs_prefix="IN: ",
            targets_prefix="OUT: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="How does the next paragraph end?\n\n{context}\n\n{options_}",
            targets="{answer}",
            inputs_prefix="Problem: ",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Write the next sentence in this paragraph:\n{context}\n{options_}",
            targets="{answer}",
            inputs_prefix="IN: ",
            targets_prefix="OUT: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Multi-choice problem: Continue writing the next sentence for the following:\n\n{context}\n\n{options_}",
            targets="{answer}",
            inputs_prefix="Question:\n",
            targets_prefix="Answer:\n",
            x_y_delimiter="\n****\n",
            example_separator="\n\n\n"),
    ],
    "piqa": [
        FewShotPattern(
            inputs="What kind of task would test someone's ability to perform "
            "physical reasoning? {options_}",
            targets="{goal}",
            inputs_prefix="Problem: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n",
            example_separator="\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Here is a goal: {goal}\n\nHow would you accomplish this "
            "goal?\n\n{options_}",
            targets="{answer}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{goal}\n{options_}",
            targets="{answer}",
            inputs_prefix="Goal:\n",
            targets_prefix="Answer:\n",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Goal: {goal}\n\nWhich of the following methods is more "
            "reasonable for accomplishing this goal?\n\n{options_}",
            targets="{answer}",
            inputs_prefix="Question:\n",
            targets_prefix="Answer:\n",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="What kind of task would test someone's ability to perform physical reasoning?",
            targets="{goal}",
            inputs_prefix="Question:\n",
            targets_prefix="Answer:\n",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="What's an example of a task that requires knowledge of physical objects to perform?",
            targets="{goal}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{goal}\n\n{options_}",
            targets="{answer}",
            inputs_prefix="Goal and options: ",
            targets_prefix="output answer: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Which of the following solutions is better for the following goal:\n{goal}\n\n{options_}",
            targets="{answer}",
            inputs_prefix="[Q]: ",
            targets_prefix="[A]: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{goal}\n\n{options_}",
            targets="{answer}",
            inputs_prefix="input question: ",
            targets_prefix="output answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Objective: {goal} {options_}",
            targets="{answer}",
            inputs_prefix="[Q]: ",
            targets_prefix="[A]: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
    ],
    "openbookqa": [
        FewShotPattern(
            inputs="Fact: {fact}\nQuestion: {question}\n{options_}",
            targets="{answer}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="What sentence would provide a factual answer "
            "to this question: \"{question}\"",
            targets="{fact}",
            inputs_prefix="question: ",
            targets_prefix="answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="What is a random fact?",
            targets="{fact}",
            inputs_prefix="Problem: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Generate a sentence that contains a fact.",
            targets="{fact}",
            inputs_prefix="Problem: ",
            targets_prefix="",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Generate a sentence that contains a fact.",
            targets="{fact}",
            inputs_prefix="Problem: ",
            targets_prefix="Sentence: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Randomly generate a fact.",
            targets="{fact}",
            inputs_prefix="Problem: ",
            targets_prefix="Random fact: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Generate a sentence that answers this question: \"{question}\".",
            targets="{fact}",
            inputs_prefix="",
            targets_prefix="Answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{fact}\nQuestion: {question}\n\nWhat's the answer? {options_}",
            targets="{answer}",
            inputs_prefix="fact and question: ",
            targets_prefix="answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Use evidence from the fact that {fact} to answer the following question. {options_} \"{question}\"",
            targets="{answer}",
            inputs_prefix="Question:\n",
            targets_prefix="Answer:\n",
            x_y_delimiter="\n\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Given the fact \"{fact}\", what is the answer to \"{question}\"\n\n{options_}",
            targets="{answer}",
            inputs_prefix="question: ",
            targets_prefix="answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            in_template_mix=False),
    ],
    "bigbench:simple_arithmetic_json.gen.blueridge_vocab.0_shot.30_examples": [
        FewShotPattern(
            inputs="{inputs}",
            targets="{targets}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{inputs}",
            targets="{targets}",
            inputs_prefix="Problem: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n",
            example_separator="\n"),
        FewShotPattern(
            inputs="{inputs}",
            targets="{targets}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n"),
        FewShotPattern(
            inputs="What is the value of {inputs}?",
            targets="{targets}",
            inputs_prefix="Problem: ",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{inputs}",
            targets="{targets}",
            inputs_prefix="",
            targets_prefix="Answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Solve this: {inputs}",
            targets="{targets}",
            inputs_prefix="",
            targets_prefix="A: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Reply with the result:\n\n{inputs}",
            targets="{targets}",
            inputs_prefix="QUESTION: ",
            targets_prefix="ANS: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Math problem: {inputs}\nAnswer:",
            targets="{targets}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n"),
        FewShotPattern(
            inputs="{inputs}",
            targets="{targets}",
            inputs_prefix="Question:\n",
            targets_prefix="Answer:\n",
            x_y_delimiter="\n",
            example_separator="\n"),
        FewShotPattern(
            inputs="{inputs}",
            targets="{targets}",
            inputs_prefix="Question answering problem: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
    ],
    "bigbench:auto_debugging.gen.blueridge_vocab.0_shot.34_examples": [
        FewShotPattern(
            inputs="{inputs}",
            targets="{targets}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{inputs}",
            targets="{targets}",
            inputs_prefix="",
            targets_prefix="Answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{inputs}",
            targets="{targets}",
            inputs_prefix="question: ",
            targets_prefix="answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Answer the following coding question:\n{inputs}",
            targets="{targets}",
            inputs_prefix="question: ",
            targets_prefix="answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Answer the following:\n\n{inputs}",
            targets="{targets}",
            inputs_prefix="",
            targets_prefix="A: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{inputs}",
            targets="Hmm... {targets}",
            inputs_prefix="Input: ",
            targets_prefix="Output: ",
            x_y_delimiter="\n",
            example_separator="\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Given the question below, answer directly after the question ended:\n{inputs}",
            targets="{targets}",
            inputs_prefix="IN: ",
            targets_prefix="",
            x_y_delimiter="\n\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Answer the following question:\n{inputs}",
            targets="{targets}",
            inputs_prefix="input: ",
            targets_prefix="answer: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{inputs}",
            targets="{targets}",
            inputs_prefix="Problem: ",
            targets_prefix="Answer: so it's ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="See this question:\n{inputs}",
            targets="{targets}",
            inputs_prefix="",
            targets_prefix="The quick answer is:\n",
            x_y_delimiter="\n\n",
            example_separator="\n\n"),
    ],
    "bigbench:strategyqa.gen.blueridge_vocab.0_shot.1000_examples": [
        FewShotPattern(
            inputs="{inputs} answer yes or no and explain.",
            targets="{targets}",
            inputs_prefix="Problem: ",
            targets_prefix="A with explanation: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{inputs}",
            targets="{targets}",
            inputs_prefix="Problem: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{inputs}",
            targets="{targets}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{inputs}",
            targets="{targets}",
            inputs_prefix="Question: ",
            targets_prefix="Answer followed by reasoning: ",
            x_y_delimiter="\n----\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Answer + your thought for the following: {inputs}",
            targets="{targets}",
            inputs_prefix="question: ",
            targets_prefix="answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Answer yes or no first, then give the reason.\n{inputs}",
            targets="{targets}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Yes/no: {inputs}",
            targets="{targets}",
            inputs_prefix="QUES: ",
            targets_prefix="ANS: ",
            x_y_delimiter="\n",
            example_separator="\n"),
        FewShotPattern(
            inputs="Yes or no first, then explain the reason: {inputs}",
            targets="{targets}",
            inputs_prefix="",
            targets_prefix="Answer with thoughts: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Answer yes or no, then explain your answer: {inputs}",
            targets="{targets}",
            inputs_prefix="question: ",
            targets_prefix="answer: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Answer this question (ye /no):\n{inputs}",
            targets="{targets}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n\n"),
    ],
    "bigbench:sufficient_information.gen.blueridge_vocab.0_shot.39_examples": [
        FewShotPattern(
            inputs="{inputs}",
            targets="{targets}",
            inputs_prefix="Question:\n",
            targets_prefix="Answer:\n",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Answer this question or say \"I don't know\": {inputs}",
            targets="{targets}",
            inputs_prefix="Question:\n",
            targets_prefix="Answer:\n",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{inputs}",
            targets="{targets}",
            inputs_prefix="Q: ",
            targets_prefix="A (or I don't know): ",
            x_y_delimiter="\n-----\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{inputs}",
            targets="{targets}",
            inputs_prefix="question: ",
            targets_prefix="answer (or I don't know): ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="You can say \"I don't know\" for the following question, but try to answer it.\n{inputs}",
            targets="{targets}",
            inputs_prefix="QUES: ",
            targets_prefix="ANS: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Answer this question or say you don't know: {inputs}",
            targets="{targets}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{inputs}",
            targets="{targets}",
            inputs_prefix="IN: ",
            targets_prefix="OUT: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{inputs}",
            targets="{targets}",
            inputs_prefix="question: ",
            targets_prefix="answer or say you don't know: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Question: {inputs}\nAnswer:",
            targets="{targets}",
            inputs_prefix="question: ",
            targets_prefix="answer: ",
            x_y_delimiter="\n",
            example_separator="\n"),
        FewShotPattern(
            inputs="Question that might not be answerable: {inputs}.",
            targets="{targets}",
            inputs_prefix="Q: ",
            targets_prefix="Try my best to answer: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
    ],
    "lambada": [
        FewShotPattern(
            inputs="{sentence}",
            targets="{answer}",
            x_y_delimiter=" ",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{sentence}",
            targets="{answer}",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{sentence}",
            targets="{answer}",
            targets_prefix="Next word for the text above: "),
        FewShotPattern(
            inputs="{sentence} _ ... Fill the blank.",
            targets="blank is {answer}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="Q: ",
            targets_prefix="A: "),
        FewShotPattern(
            inputs="Continue writing the following text: {sentence}",
            targets="{answer}",
            inputs_prefix="TEXT: ",
            targets_prefix="CONTINUE: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="A completion problem: {sentence}",
            targets="{answer}",
            inputs_prefix="input question: ",
            targets_prefix="output answer: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Text to complete: {sentence}",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="completion:\n",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Complete this sentence: {sentence}",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="Answer: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="SENTENCE: {sentence}\nSo, what's next?",
            targets="{answer}",
            inputs_prefix="Problem: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{sentence}...",
            targets="{answer}",
            inputs_prefix="[Q]: ",
            targets_prefix="[A]: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
    ],
    "cot_gsm8k": [
        FewShotPattern(
            inputs="Answer the following question.\n{question}",
            targets="Step-by-step reasoning process: {chain_of_thought}\n"
            "The answer is {answer}.",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="{chain_of_thought} The answer is {answer}.",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="Q: ",
            targets_prefix="A: "),
        FewShotPattern(
            inputs="{question}",
            targets="Step-by-step reasoning process: {chain_of_thought}\n"
            "So the answer is {answer}.",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="Question: ",
            targets_prefix="Answer: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question} Give the rationale and then the answer.",
            targets="Let's think step by step. {chain_of_thought}. "
            "The answer is: {answer}.",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Answer this question:{question}",
            targets="{chain_of_thought}\nThe answer is {answer}.",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="{chain_of_thought} The answer is {answer}.",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="Question: ",
            targets_prefix="Answer: "),
        FewShotPattern(
            inputs="{question}",
            targets="{chain_of_thought}\nSo the answer is {answer}.",
            x_y_delimiter="\n",
            example_separator="\n\n\n",
            inputs_prefix="Question: ",
            targets_prefix="Answer with step-by-step thinking: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="{chain_of_thought}. The answer is: {answer}.",
            targets_prefix="Let's think: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="{chain_of_thought}\nSo the answer is {answer}.",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="[Question]: ",
            targets_prefix="[Answer]: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="{chain_of_thought} The answer is {answer}.",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="[Question]",
            targets_prefix="[Answer]"),
    ],
    "cot_strategyqa": [
        FewShotPattern(
            inputs="{question}",
            targets="{chain_of_thought}\nThe answer is {answer}.",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="{chain_of_thought}. The answer is: {answer}.",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="Question: ",
            targets_prefix="Answer: "),
        FewShotPattern(
            inputs="{question}",
            targets="My step-by-step reasoning: {chain_of_thought}\n"
            "So, the answer is {answer}.",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="Q--",
            targets_prefix="A--",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}\nRationale first then the answer.",
            targets="{chain_of_thought}. The answer is: {answer}.",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{question}",
            targets="{chain_of_thought}\nThe answer is {answer}.",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="Chain of thought: {chain_of_thought} "
            "The answer is {answer}.",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            inputs_prefix="[Question]: ",
            targets_prefix="[Answer]: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="My reasoning: {chain_of_thought}\nThe answer: {answer}.",
            x_y_delimiter="\n",
            example_separator="\n\n\n",
            inputs_prefix="*Q:* ",
            targets_prefix="*A:* ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}\nPlease give rationale first, then the answer.",
            targets="{chain_of_thought}. The answer is: {answer}.",
            inputs_prefix="QUESTION: ",
            targets_prefix="ANSWER: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{question}",
            targets="{answer}\nExplanation: {chain_of_thought}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="The detailed solution is: {chain_of_thought}.",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="Student question: ",
            targets_prefix="Teacher response: ",
            in_template_mix=False),
    ],
    "cot_esnli": [
        FewShotPattern(
            inputs="[QUESTION] {question}",
            targets="{chain_of_thought}\nThe answer is {answer}.",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{question}",
            targets="Let's think. {chain_of_thought} "
            "The answer is {answer}.",
            x_y_delimiter="\n",
            example_separator="\n--\n",
            inputs_prefix="Next Question: ",
            targets_prefix="My Answer: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="Let's solve this gradually. {chain_of_thought}\n"
            "Answer is {answer}.",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            inputs_prefix="QUESTION: ",
            targets_prefix="SOLUTION: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="{chain_of_thought}. The answer is: {answer}.",
            x_y_delimiter="\n--\n",
            example_separator="\n----\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="{chain_of_thought}\nThe answer is {answer}.",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{question}",
            targets="Let's think. {chain_of_thought} The answer is {answer}.",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="Next Question: ",
            targets_prefix="My Answer: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="[Q] {question}",
            targets="[A] {chain_of_thought}\nThe answer is {answer}.",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="Let's think. {chain_of_thought} The answer is {answer}.",
            x_y_delimiter="\n",
            example_separator="\n\n\n",
            inputs_prefix="Student asked: ",
            targets_prefix="Teacher's response: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="{chain_of_thought}\nThe answer is {answer}.",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            inputs_prefix="QUESTION: ",
            targets_prefix="Let's solve it slowly: "),
        FewShotPattern(
            inputs="{question}",
            targets="{answer}\nExplanation: {chain_of_thought}.",
            x_y_delimiter="\n\n",
            example_separator="\n\n",
            in_template_mix=False),
    ],
    "stream_aqua": [
        FewShotPattern(
            inputs="{question}",
            targets="OK... Stream of consciousness: {chain_of_thought}\n"
            "The answer is {answer}.",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="Stream of consciousness: {chain_of_thought} "
            "The answer is {answer}.",
            x_y_delimiter="\n",
            example_separator="\n--\n",
            inputs_prefix="q: ",
            targets_prefix="a: "),
        FewShotPattern(
            inputs="{question}",
            targets="{answer}\nStream of consciousness: {chain_of_thought}",
            x_y_delimiter="\n",
            example_separator="\n\n\n",
            inputs_prefix="",
            targets_prefix="Answer and stream of consciousness: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="{chain_of_thought} So the answer is: {answer}.",
            x_y_delimiter="\n--\n",
            example_separator="\n-----\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="{chain_of_thought}\nThe answer is {answer}.",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="Question: ",
            targets_prefix="Answer: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="{chain_of_thought} The answer is {answer}.",
            x_y_delimiter="\n",
            example_separator="\n--\n",
            inputs_prefix="q: ",
            targets_prefix="a: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="{answer}\nStream of consciousness:{chain_of_thought}",
            x_y_delimiter="\n",
            example_separator="\n\n\n",
            inputs_prefix="question:",
            targets_prefix="answer:",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="{chain_of_thought} So the answer is: {answer}.",
            x_y_delimiter="\n\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="OK... {chain_of_thought}\nThe answer is {answer}.",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="q: ",
            targets_prefix="a: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="Stream of consciousness: {chain_of_thought} "
            "The answer is {answer}.",
            x_y_delimiter="\n",
            example_separator="\n--\n",
            inputs_prefix="question: ",
            targets_prefix="answer: "),
    ],
    "cot_creak": [
        FewShotPattern(
            inputs="{question}",
            targets="The answer is {answer}.\n"
            "Chain of thoughts: {chain_of_thought}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="I'm thinking hard. So here's my take: {chain_of_thought} "
            "The answer is {answer}.",
            x_y_delimiter="\n",
            example_separator="\n---\n",
            inputs_prefix="Ques: ",
            targets_prefix="Ans: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="{answer}\n{chain_of_thought}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="",
            targets_prefix=""),
        FewShotPattern(
            inputs="{question}",
            targets="Let me think out loud. {chain_of_thought} "
            "The answer is {answer}.",
            x_y_delimiter="\n\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="The answer is {answer}.\n"
            "Explanation: {chain_of_thought}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="Question: ",
            targets_prefix="Ans and explanation: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="Oh man, I think this is the solution: {chain_of_thought} "
            "The answer is {answer}.",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="Question part\n",
            targets_prefix="Answer part\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="{answer}\n{chain_of_thought}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="**Q**\n",
            targets_prefix="**A**\n"),
        FewShotPattern(
            inputs="Question: {question}",
            targets="Let me think..... {chain_of_thought} "
            "The answer is {answer}.",
            x_y_delimiter="\n\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="The answer is {answer}.\n"
            "Chain of thoughts: {chain_of_thought}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="Answer the following question: ",
            targets_prefix="My answer and thoughts: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="Here's my solution: {chain_of_thought} "
            "The answer is {answer}.",
            x_y_delimiter="\n",
            example_separator="\n****\n",
            inputs_prefix="[Ques]: ",
            targets_prefix="[Ans]: ",
            in_template_mix=False),
    ],
    "cot_ecqa": [
        FewShotPattern(
            inputs="{question}",
            targets="The answer is {answer}\n"
            "CoT: {chain_of_thought}.",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="QUESTION: ",
            targets_prefix="ME: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="Let me think step-by-step: {chain_of_thought} "
            "The answer is {answer}.",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="{chain_of_thought}\n{answer}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="",
            targets_prefix="",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="Let's do it gradually: {chain_of_thought}... "
            "So the answer is {answer}.",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="{chain_of_thought}\nThe answer is {answer}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="QUESTION: ",
            targets_prefix="ANSWER: "),
        FewShotPattern(
            inputs="{question}",
            targets="Let me think step-by-step: {chain_of_thought} "
            "So the answer must be {answer}.",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="QUESTION: ",
            targets_prefix="ANSWER: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="{chain_of_thought}\nThe answer is {answer}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="q: ",
            targets_prefix="a: "),
        FewShotPattern(
            inputs="{question}",
            targets="Let's solve it slow. {chain_of_thought}... "
            "So the answer is {answer}.",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="The answer is {answer}\nExplanation: {chain_of_thought}.",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="QUESTION: ",
            targets_prefix="ANSWER W/ DETAILS: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="Let me think. {chain_of_thought} "
            "The answer is {answer}.",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="Person A: ",
            targets_prefix="Person B: ",
            in_template_mix=False),
    ],
    "cot_sensemaking": [
        FewShotPattern(
            inputs="{question}",
            targets="{chain_of_thought}\nThe answer is {answer}.",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="Teacher: ",
            targets_prefix="Student: "),
        FewShotPattern(
            inputs="{question}",
            targets="Chain of thought: {chain_of_thought} "
            "The answer is {answer}.",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="Jax: ",
            targets_prefix="Alex: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="Let's see... {chain_of_thought}\n{answer}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="Ques:",
            targets_prefix="Ans:",
            in_template_mix=False),
        FewShotPattern(
            inputs="[{question}]",
            targets="My step-by-step solution: {chain_of_thought}... "
            "So the answer is [{answer}]",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="{chain_of_thought}\nThe answer is {answer}.",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="question in book:\n",
            targets_prefix="standard solution:\n"),
        FewShotPattern(
            inputs="{question}",
            targets="This should be the solution: {chain_of_thought} "
            "The answer is {answer}.",
            x_y_delimiter="\n",
            example_separator="\n\n\n",
            inputs_prefix="Jade: ",
            targets_prefix="Lux: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="{chain_of_thought}\n[{answer}]",
            x_y_delimiter="\n",
            example_separator="\n\n\n\n",
            inputs_prefix="Q:",
            targets_prefix="A:",
            in_template_mix=False),
        FewShotPattern(
            inputs="[{question}]",
            targets="My step-by-step solution first: {chain_of_thought}... "
            "The answer is [{answer}]",
            x_y_delimiter="\n",
            example_separator="\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="{chain_of_thought}\nThe answer is {answer}.",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="[TEACHER] ",
            targets_prefix="[Student] "),
        FewShotPattern(
            inputs="{question}",
            targets="Thoughts: {chain_of_thought} "
            "The answer is [{answer}]",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="q: ",
            targets_prefix="a: ",
            in_template_mix=False),
    ],
    "cot_qasc": [
        FewShotPattern(
            inputs="{question}",
            targets="{chain_of_thought}\nThe answer is {answer}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="question: ",
            targets_prefix="answer: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="So, my chain of thought: {chain_of_thought} "
            "The answer is {answer}.",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="[{question}]",
            targets="[{chain_of_thought}\n{answer}]",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="Ques:",
            targets_prefix="Ans:",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="Let's think first: {chain_of_thought}... "
            "So the answer is [{answer}]",
            x_y_delimiter="\n--\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{question}",
            targets="{chain_of_thought}\nThe answer is {answer}",
            x_y_delimiter="\n",
            example_separator="\n\n\n",
            inputs_prefix="(question). ",
            targets_prefix="(answer). ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="{answer}... Explanation: {chain_of_thought} "
            "That's why the answer is {answer}.",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="q: ",
            targets_prefix="a: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="[{chain_of_thought}]\n[{answer}]",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="[[Ques]]: ",
            targets_prefix="[[Ans]]: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="Let's think first: {chain_of_thought}... "
            "So the answer is [{answer}]",
            x_y_delimiter="\n--\n",
            example_separator="\n------\n"),
        FewShotPattern(
            inputs="{question}",
            targets="{chain_of_thought}\nThe answer is {answer}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="question... ",
            targets_prefix="answer... ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="Here's my solution: {chain_of_thought} "
            "The answer is {answer}.",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="[Q] ",
            targets_prefix="[A] ",
            in_template_mix=False),
    ],
    "stream_qed": [
        FewShotPattern(
            inputs="{question}",
            targets="{chain_of_thought}\nThe answer is {answer}",
            x_y_delimiter="\n",
            example_separator="\n\n\n",
            inputs_prefix="q... ",
            targets_prefix="a... ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="{chain_of_thought} The answer is {answer}.",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="Quick Question: ",
            targets_prefix="My answer: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="OK... {chain_of_thought}\n{answer}.",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="Student A:",
            targets_prefix="Student B:"),
        FewShotPattern(
            inputs="{question} Let's do a good job answering this.",
            targets="Stream of consciousness: {chain_of_thought}... "
            "The answer is {answer}.",
            x_y_delimiter="\n--\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="{chain_of_thought}\nSo the answer must be {answer}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="## question\n",
            targets_prefix="## answer\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="{answer}. How to explain the answer? {chain_of_thought}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="OK... {chain_of_thought}\n{answer}.",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="Problem:",
            targets_prefix="Solution:"),
        FewShotPattern(
            inputs="Answer this question please:\n{question}",
            targets="Stream of random thoughts: {chain_of_thought}... "
            "The answer is {answer}.",
            x_y_delimiter="\n\n",
            example_separator="\n----\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="{chain_of_thought}\nFINAL ANSWER: {answer}",
            x_y_delimiter="\n",
            example_separator="\n\n\n",
            inputs_prefix="# QUESTION\n",
            targets_prefix="# ANSWER\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="{chain_of_thought} The answer is {answer}.",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="Question: ",
            targets_prefix="Answer: ",
            in_template_mix=False),
    ],
    "cot_input_inversion_gsm8k": [
        FewShotPattern(
            inputs="Consider the Q and A. Q: {question}\nA: {answer}\nWhat is the step-by-step reasoning process?",
            targets="{chain_of_thought}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            targets_prefix="Step-by-step reasoning process: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{chain_of_thought}\nThe answer: {answer}\nWhat was the question?",
            targets="{question}",
            inputs_prefix="Reasoning and answer: ",
            targets_prefix="Question: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Come up with a question and reasoning that would justify this answer: {answer}",
            targets="The question is: {question}\nStep-by-step reasoning process: {chain_of_thought}",
            targets_prefix="Question and rationale: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Step-by-step reasoning process: {chain_of_thought}\nThe question and answer:",
            targets="{question}\nThe answer is {answer}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            targets_prefix="Question and answer: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="Q: {question}\nA: {answer}",
            targets="Step-by-step reasoning process: {chain_of_thought}",
            x_y_delimiter="\n",
            example_separator="\n\n\n",
            targets_prefix="",
            in_template_mix=False),
        FewShotPattern(
            inputs="CoT: {chain_of_thought}\nThe answer: {answer}",
            targets="{question}",
            inputs_prefix="Reasoning & answer: ",
            targets_prefix="Question: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Come up with a question and reasoning that would justify [{answer}] as the answer.",
            targets="The question is: {question}\nReasoning: {chain_of_thought}",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Step-by-step reasoning process: {chain_of_thought}",
            targets="The question is {question}\nThe answer is {answer}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            targets_prefix="[Q & A] ",
            in_template_mix=False),
        FewShotPattern(
            inputs="We have a question: {question}\nAnd an answer: {answer}\nSo how you got the answer?",
            targets="{chain_of_thought}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            targets_prefix="",
            in_template_mix=False),
        FewShotPattern(
            inputs="{chain_of_thought}\nThe answer: {answer}",
            targets="{question}",
            inputs_prefix="",
            targets_prefix="Reverse engineering the question: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
    ],
    "cot_input_inversion_strategyqa": [
        FewShotPattern(
            inputs="{chain_of_thought}\n{answer}",
            targets="{question}",
            inputs_prefix="Reasoning & answer: ",
            targets_prefix="Question: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{answer}",
            targets="The question is: {question}\n{chain_of_thought}",
            targets_prefix="Question and rationale: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Step-by-step reasoning process: {chain_of_thought}\nThe question and answer:",
            targets="{question}\n{answer}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="",
            targets_prefix="Question and answer: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="Q: {question}\nA: {answer}",
            targets="{chain_of_thought}",
            inputs_prefix="",
            targets_prefix="CoT: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{chain_of_thought}\n{answer}",
            targets="{question}",
            inputs_prefix="CoT and answer: ",
            targets_prefix="Do reverse engineering and find the question: ",
            x_y_delimiter="\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{answer}",
            targets="{question}\n{chain_of_thought}",
            inputs_prefix="Known answer: ",
            targets_prefix="Now, what could be the question and solution? ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Step-by-step reasoning process: {chain_of_thought}",
            targets="{question}\n{answer}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="",
            targets_prefix="[Q and A]: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="Q: {question}\nA: {answer}",
            targets="{chain_of_thought}",
            inputs_prefix="",
            targets_prefix="Explanation: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Solution: {chain_of_thought}\nAnswer: {answer}",
            targets="{question}",
            inputs_prefix="",
            targets_prefix="Question: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{answer}",
            targets="The question is: {question}\nThe rationale is: {chain_of_thought}",
            inputs_prefix="The answer: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
    ],
    "cot_input_inversion_esnli": [
        FewShotPattern(
            inputs="Q: {question}\nA: {answer}",
            targets="{chain_of_thought}",
            inputs_prefix="",
            targets_prefix="Chain-of-thought: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{chain_of_thought}\nThe question and answer are below.",
            targets="{question}\n{answer}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="",
            targets_prefix=""),
        FewShotPattern(
            inputs="{chain_of_thought}\n{answer}",
            targets="{question}",
            inputs_prefix="R & A: ",
            targets_prefix="Q: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{answer}",
            targets="{question}\n[CoT] {chain_of_thought}\n",
            inputs_prefix="[Ans] ",
            targets_prefix="[Question] ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Q: {question}\nA: {answer}",
            targets="{chain_of_thought}",
            inputs_prefix="",
            targets_prefix="CoT: ",
            x_y_delimiter="\n",
            example_separator="\n****\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{chain_of_thought}\nThe question and answer are below.",
            targets="{question}\n{answer}",
            x_y_delimiter="\n",
            example_separator="\n\n*****\n\n",
            inputs_prefix="",
            targets_prefix=""),
        FewShotPattern(
            inputs="{chain_of_thought}\n{answer}",
            targets="{question}",
            inputs_prefix="Reasoning & Answer: ",
            targets_prefix="Question: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{answer}",
            targets="{question}\n*CoT* {chain_of_thought}",
            inputs_prefix="*Ans* ",
            targets_prefix="*Question* ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Q: {question}\nA: {answer}",
            targets="{chain_of_thought}",
            inputs_prefix="Question and answer: ",
            targets_prefix="Explanation: ",
            x_y_delimiter="\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{chain_of_thought}. So what could be the question?",
            targets="{question}\n{answer}",
            x_y_delimiter="\n",
            example_separator="\n\n\n",
            inputs_prefix="",
            targets_prefix="Question followed by answer: "),
    ],
    "stream_input_inversion_aqua": [
        FewShotPattern(
            inputs="Ques: {question}\nAns: {answer}",
            targets="CoT: {chain_of_thought}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{answer}",
            targets="{question}\n[CoT] {chain_of_thought}",
            inputs_prefix="[Ans] ",
            targets_prefix="[Question] ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Reasoning: {chain_of_thought}\nAns: {answer}",
            targets="Question: {question}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{chain_of_thought}",
            targets="{question}\n{answer}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="",
            targets_prefix="",
            in_template_mix=False),
        FewShotPattern(
            inputs="[Ques]: {question}\n*Ans*: {answer}",
            targets="--CoT--: {chain_of_thought}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{answer}",
            targets="{question}\n[CoT]: [{chain_of_thought}]",
            inputs_prefix="Answer: ",
            targets_prefix="Question: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Rationale: {chain_of_thought}\nThe answer: {answer}",
            targets="Question: {question}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{chain_of_thought}",
            targets="{question}\n{answer}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="## Solution\n",
            targets_prefix="## What the question and answer could be\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Question: {question}\nAns: {answer}",
            targets="CoT: {chain_of_thought}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n----\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{answer}",
            targets="{question}\n(CoT). {chain_of_thought}",
            inputs_prefix="(Ans). ",
            targets_prefix="(Question). ",
            x_y_delimiter="\n",
            example_separator="\n\n\n",
            in_template_mix=False),
    ],
    "cot_input_inversion_creak": [
        FewShotPattern(
            inputs="{question}\n{answer}",
            targets="{chain_of_thought}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{chain_of_thought}\n{answer}",
            targets="{question}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{chain_of_thought}",
            targets="{question}\n{answer}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="",
            targets_prefix="",
            in_template_mix=False),
        FewShotPattern(
            inputs="{answer}",
            targets="{question}\n*CoT* {chain_of_thought}\n",
            inputs_prefix="*Ans* ",
            targets_prefix="*Question* ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}\n{answer}",
            targets="{chain_of_thought}",
            inputs_prefix="Q&A: ",
            targets_prefix="Exp: ",
            x_y_delimiter="\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{chain_of_thought}\n{answer}",
            targets="{question}",
            inputs_prefix="Explanation and answer: ",
            targets_prefix="The corresponding question: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{chain_of_thought}",
            targets="[{question}]\n[{answer}]",
            x_y_delimiter="\n",
            example_separator="\n\n\n",
            inputs_prefix="Idea: ",
            targets_prefix="Generated [question] and [answer]: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{answer}",
            targets="{question}\nCoT: {chain_of_thought}\n",
            inputs_prefix="Ans: ",
            targets_prefix="Question: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Q: {question}\nA: {answer}",
            targets="How to explain the answer: {chain_of_thought}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="CoT: {chain_of_thought}\nAnswer: {answer}",
            targets="What is the question? This is the question: {question}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
    ],
    "cot_input_inversion_ecqa": [
        FewShotPattern(
            inputs="{chain_of_thought}\n{answer}",
            targets="{question}",
            inputs_prefix="** ",
            targets_prefix="** ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{chain_of_thought}",
            targets="{question}\n{answer}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}\n{answer}",
            targets="{chain_of_thought}",
            inputs_prefix="[1] ",
            targets_prefix="[2] ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="*Ans* {answer}",
            targets="*Question* {question}\n*CoT* {chain_of_thought}\n",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{chain_of_thought}\n{answer}",
            targets="{question}",
            inputs_prefix="Detailed logic: ",
            targets_prefix="Question for this logic: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{chain_of_thought}",
            targets="{question}\n{answer}",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            inputs_prefix="CoT: ",
            targets_prefix="Q&A: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}\n{answer}",
            targets="{chain_of_thought}",
            inputs_prefix="## Question and Answer ",
            targets_prefix="## Chain-of-thought ",
            x_y_delimiter="\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs=">Ans< {answer}",
            targets=">Question< {question}\n>CoT< {chain_of_thought}\n",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n--\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{chain_of_thought}\n{answer}",
            targets="{question}",
            inputs_prefix="Logic ==> ",
            targets_prefix="Question ==> ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{chain_of_thought}",
            targets="{question}\n{answer}",
            x_y_delimiter="\n",
            example_separator="\n\n\n",
            inputs_prefix="Q:\n",
            targets_prefix="A:\n",
            in_template_mix=False),
    ],
    "cot_input_inversion_sensemaking": [
        FewShotPattern(
            inputs="*Ans* {answer}",
            targets="*Question* {question}\n*CoT* {chain_of_thought}\n",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n****\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{chain_of_thought}",
            targets="Question: {question}\nAnswer: {answer}",
            x_y_delimiter="\n--\n",
            example_separator="\n\n\n",
            inputs_prefix="Chain-of-thought: ",
            targets_prefix=""),
        FewShotPattern(
            inputs="[{chain_of_thought}]\n[{answer}]",
            targets="[{question}]",
            inputs_prefix="- ",
            targets_prefix="+ ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}\n{answer}",
            targets="{chain_of_thought}",
            inputs_prefix="Question and Answer: ",
            targets_prefix="Some stream of consciousness: ",
            x_y_delimiter="\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Answer -> {answer}",
            targets="Question -> {question}\nRationale -> {chain_of_thought}\n",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{chain_of_thought}",
            targets="Question: {question}\nAnswer: {answer}",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            inputs_prefix="Chain-of-thought: ",
            targets_prefix=""),
        FewShotPattern(
            inputs="<{chain_of_thought}>\n<{answer}>",
            targets="<{question}>",
            inputs_prefix="Idea: ",
            targets_prefix="Generated question: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}\n{answer}",
            targets="{chain_of_thought}",
            inputs_prefix="Question and Answer: ",
            targets_prefix="Some idea for the solution: ",
            x_y_delimiter="\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Answer: [{answer}]",
            targets="Question: [{question}]\nSolution: [{chain_of_thought}]",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{chain_of_thought}",
            targets="Question: {question}\nAnswer: {answer}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="Explanation for the following question's answer: ",
            targets_prefix=""),
    ],
    "cot_input_inversion_qasc": [
        FewShotPattern(
            inputs="{question}\n{answer}",
            targets="{chain_of_thought}",
            inputs_prefix="Ques and Ans: ",
            targets_prefix="Logic chain: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{chain_of_thought}",
            targets="Q: {question}\nA: {answer}",
            x_y_delimiter="\n",
            example_separator="\n--\n",
            inputs_prefix="Chain-of-thought: ",
            targets_prefix="",
            in_template_mix=False),
        FewShotPattern(
            inputs="*{chain_of_thought}*\n*{answer}*",
            targets="*{question}*",
            inputs_prefix="Line 1: ",
            targets_prefix="Line 2: ",
            x_y_delimiter="\n",
            example_separator="\n--\n"),
        FewShotPattern(
            inputs="Ans: {answer}",
            targets="Question: {question}\nCoT: {chain_of_thought}\n",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}\n{answer}",
            targets="{chain_of_thought}",
            inputs_prefix="Ques and Ans: ",
            targets_prefix="Logic for the answer: ",
            x_y_delimiter="\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{chain_of_thought}",
            targets="Q: {question}\nA: {answer}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="Solution idea: ",
            targets_prefix="",
            in_template_mix=False),
        FewShotPattern(
            inputs="*{chain_of_thought}*\n*{answer}*",
            targets="*{question}*",
            inputs_prefix="Logic of a solution: ",
            targets_prefix="The original question: ",
            x_y_delimiter="\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Ans: {answer}",
            targets="Question: {question}\nCoT: {chain_of_thought}\n",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}\n{answer}",
            targets="[[{chain_of_thought}]]",
            inputs_prefix="Ques and Ans: ",
            targets_prefix="Explanation for the Ans above: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{chain_of_thought}",
            targets="Q: {question}\nA: {answer}",
            x_y_delimiter="\n",
            example_separator="\n--\n",
            inputs_prefix="Logic for the Q&A below: ",
            targets_prefix="",
            in_template_mix=False),
    ],
    "stream_input_inversion_qed": [
        FewShotPattern(
            inputs="Ans: {answer}",
            targets="Ques: {question}\nCoT: {chain_of_thought}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{chain_of_thought}",
            targets="Q: {question}\nA: {answer}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="Stream of consciousness: ",
            targets_prefix="",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}\n{answer}",
            targets="{chain_of_thought}",
            inputs_prefix="Ques & Ans: ",
            targets_prefix="Stream of consciousness: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{chain_of_thought}\n{answer}",
            targets="{question}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Answer: {answer}. Now, what could be the question and solution-maybe?",
            targets="Ques: {question}\nCoT: {chain_of_thought}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n---\n"),
        FewShotPattern(
            inputs="Idea for the Q&A below: {chain_of_thought}",
            targets="Q: {question}\nA: {answer}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="",
            targets_prefix="",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}\n{answer}",
            targets="{chain_of_thought}",
            inputs_prefix="Ques & Ans: ",
            targets_prefix="Stream of consciousness: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{chain_of_thought}\n{answer}",
            targets="{question}",
            inputs_prefix="a: ",
            targets_prefix="q: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Ans: {answer}",
            targets="Ques: {question}\nCoT: {chain_of_thought}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n====\n"),
        FewShotPattern(
            inputs="{chain_of_thought}",
            targets="Q: {question}\nA: {answer}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="Some random thoughts: ",
            targets_prefix="Generated quizz based on thoughts",
            in_template_mix=False),
    ],
    "strategyqa": [
        FewShotPattern(
            inputs="Question - {question}",
            targets="Answer - {answer}",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(inputs="{question}", targets="{answer}"),
        FewShotPattern(
            inputs="{question}",
            targets="{answer}",
            inputs_prefix="Yes or no: ",
            targets_prefix="The answer is: "),
        FewShotPattern(
            inputs="{question} Answer yes or no.",
            targets="{answer}",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{question}",
            targets="{answer}",
            inputs_prefix="[Q]: ",
            targets_prefix="[A]: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Reply yes or no: {question}",
            targets="{answer}",
            inputs_prefix="[Q]: ",
            targets_prefix="[A]: ",
            x_y_delimiter="\n++++++++++\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{question}",
            targets="{answer}",
            inputs_prefix="Question: ",
            targets_prefix="Yes or no: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{question}\nYes or No?",
            targets="{answer}",
            inputs_prefix="input question: ",
            targets_prefix="output answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{question} Answer yes or no.",
            targets="{answer}",
            inputs_prefix="Question: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{question}",
            targets="{answer}",
            inputs_prefix="IN QUESTION: ",
            targets_prefix="OUT ANS: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
    ],
    "unified_qa_science_inst": [
        FewShotPattern(
            inputs="{question} {options_}",
            targets="{answer}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(inputs="{question}\n{options_}", targets="{answer}"),
        FewShotPattern(
            inputs="Q: {question} ---- {options_}", targets="A: {answer}"),
        FewShotPattern(
            inputs="{question}\n===\n{options_}\n",
            targets="{answer}",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{question}\n{options_}",
            targets="{answer}",
            inputs_prefix="Question:\n",
            targets_prefix="Answer:\n",
            x_y_delimiter="\n",
            example_separator="\n"),
        FewShotPattern(
            inputs="Answer this question:\n{question}\n{options_}",
            targets="{answer}",
            inputs_prefix="Input: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n",
            example_separator="\n"),
        FewShotPattern(
            inputs="{question} {options_}",
            targets="{answer}",
            inputs_prefix="*Question*: ",
            targets_prefix="*Answer*: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{question}\n{options_}",
            targets="{answer}",
            inputs_prefix="question:\n",
            targets_prefix="answer:\n",
            x_y_delimiter="\n\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{question}\n===\n{options_}",
            targets="{answer}",
            inputs_prefix="[Q]: ",
            targets_prefix="[A]: ",
            x_y_delimiter="\n****\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{question}\n{options_}",
            targets="{answer}",
            inputs_prefix="question below:\n",
            targets_prefix="answer below:\n",
            x_y_delimiter="\n",
            example_separator="\n\n"),
    ],
    "predict_next_turn_dialog": [
        FewShotPattern(
            inputs="{dialog_} ",
            targets="{answer}",
            inputs_prefix="Example conversation: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Read the dialog and predict the next turn. {dialog_}",
            targets="{answer}"),
        FewShotPattern(
            inputs="Write the response (start with \"Response:\") {dialog_}",
            targets="Response: {answer}",
            inputs_prefix="Example conversation: "),
        FewShotPattern(
            inputs="See the conversation examples, and predict the next turn. "
            "{dialog_}",
            targets="{answer}",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Get response for this dialogue: {dialog_}",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n"),
        FewShotPattern(
            inputs="{dialog_}",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="Next turn: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="What could be the response? {dialog_}",
            targets="{answer}",
            inputs_prefix="Problem: ",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Write another turn of this conversation. {dialog_}",
            targets="{answer}",
            inputs_prefix="QUESTION: ",
            targets_prefix="ANS: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Write a response. {dialog_}",
            targets="{answer}",
            inputs_prefix="question: ",
            targets_prefix="response: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="See the conversation. {dialog_}",
            targets="{answer}",
            inputs_prefix="Q: ",
            targets_prefix="Next: ",
            x_y_delimiter="\n****\n",
            example_separator="\n\n\n"),
    ],
    "t0_question_answer": [
        FewShotPattern(
            inputs="{question}",
            targets="Ans: {answer}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="{answer}",
            targets_prefix="Answer: ",
            x_y_delimiter="\n----\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{question}",
            targets="{answer}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{question}",
            targets="{answer}",
            inputs_prefix="Question: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{question}",
            targets="{answer}",
            inputs_prefix="[Q]: ",
            targets_prefix="[A]: ",
            x_y_delimiter="\n****\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Please answer the following: {question}",
            targets="{answer}",
            inputs_prefix="input: ",
            targets_prefix="output: ",
            x_y_delimiter="\n++++++++++\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Please answer this: {question}",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="Answer: ",
            x_y_delimiter="\n++++++++\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{question}",
            targets="{answer}",
            inputs_prefix="Problem: ",
            targets_prefix="A: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Given the question: {question}",
            targets="{answer}",
            inputs_prefix="Problem: ",
            targets_prefix="The answer is:\n",
            x_y_delimiter="\n++++++++++++++++++++++++++++++++\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{question}???",
            targets="{answer}",
            inputs_prefix="input question: ",
            targets_prefix="output answer: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
    ],
    "t0_multiple_choice": [
        FewShotPattern(
            inputs="{question}",
            targets="{answer}",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{question}",
            targets="{answer}",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Q: {question}",
            targets="A: {answer}",
            x_y_delimiter="\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{question}",
            targets="{answer}",
            inputs_prefix="Question: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{question}",
            targets="{answer}",
            inputs_prefix="*Question*\n",
            targets_prefix="**Answer**\n",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{question}",
            targets="{answer}",
            x_y_delimiter="\n",
            example_separator="\n------\n"),
        FewShotPattern(
            inputs="{question}",
            targets="{answer}",
            inputs_prefix="(Question)\n",
            targets_prefix="(Answer)\n",
            x_y_delimiter="\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{question}",
            targets="{answer}",
            inputs_prefix="Ques: ",
            targets_prefix="Ans: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{question}",
            targets="{answer}",
            inputs_prefix="(Q).\n",
            targets_prefix="(A).\n",
            x_y_delimiter="\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{question}",
            targets="{answer}",
            inputs_prefix="Ques:",
            targets_prefix="Ans:",
            x_y_delimiter="\n\n",
            example_separator="\n-----\n"),
    ],
    "t0_multiple_choice_separated_options": [
        FewShotPattern(
            inputs="{question}\n{options_}",
            targets="{answer}",
            x_y_delimiter="\n",
            example_separator="\n--\n"),
        FewShotPattern(
            inputs="{question}\n{options_}",
            targets="{answer}",
            targets_prefix="Answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{question} | {options_}",
            targets="A: {answer}",
            inputs_prefix="Q: ",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}\n{options_}",
            targets="{answer}",
            inputs_prefix="Question: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{question}\n{options_}",
            targets="{answer}",
            inputs_prefix="input with options: ",
            targets_prefix="output: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Q: {question}\n\n{options_}",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="A: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{options_} Now, answer this question: {question}\nA:",
            targets="{answer}",
            inputs_prefix="input: ",
            targets_prefix="output: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{question}\n\n{options_}",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="Answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{options_}\nQ: {question}",
            targets="{answer}",
            inputs_prefix="Problem: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n",
            example_separator="\n"),
        FewShotPattern(
            inputs="{options_}\n\n{question}",
            targets="{answer}",
            inputs_prefix="Problem:",
            targets_prefix="A: ",
            x_y_delimiter="\n****\n",
            example_separator="\n\n\n"),
    ],
    "program_synthesis_dmcc_python": [
        FewShotPattern(
            inputs="{question}",
            targets="{answer}",
            inputs_prefix="Example Question: ",
            targets_prefix="Example Solution: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Write a program that answers the question. {question}",
            targets="{answer}",
            inputs_prefix="Example Question: ",
            targets_prefix="Example Solution: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Solve this problem with code.\n{question}",
            targets="{answer}",
            inputs_prefix="Ex Question: ",
            targets_prefix="Ex Solution: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="See a series of task descriptions and coded solutions.\n{question}",
            targets="{answer}",
            inputs_prefix="Question: ",
            targets_prefix="Coded Solution: ",
            x_y_delimiter="\n ---- \n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="[code]{question}[BEGIN]",
            targets="{answer}[DONE]",
            inputs_prefix="[Q]: ",
            targets_prefix="[A]: ",
            x_y_delimiter="\n++++++++++\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="{answer}",
            inputs_prefix="INPUT CODING QUESTION: ",
            targets_prefix="OUTPUT ANSWER: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{question}",
            targets="{answer}",
            inputs_prefix="To solve: ",
            targets_prefix="Code solution in Python is:\n",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Coding Problem.\n{question}",
            targets="{answer}",
            inputs_prefix="input question: ",
            targets_prefix="output answer: ",
            x_y_delimiter="\n****\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{question}",
            targets="{answer}",
            inputs_prefix="QUESTION: ",
            targets_prefix="CODE: ",
            x_y_delimiter="\n",
            example_separator="\n"),
        FewShotPattern(
            inputs="Solve this problem. {question}",
            targets="{answer}",
            inputs_prefix="QUES: ",
            targets_prefix="ANS: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n"),
    ],
    "program_synthesis_dr_repair": [
        FewShotPattern(
            inputs="Fix the broken code:\n{question}",
            targets="{answer}",
            targets_prefix="Fixed Code: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Examples of incorrect and correct code. ```{question}```",
            targets="{answer}```",
            targets_prefix="Correct code: ```",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Fix this program.\n{question}",
            targets="{answer}",
            targets_prefix="Ex solution: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Come up with solutions for the broken code examples.\n{question}",
            targets="{answer}",
            targets_prefix="Fixed: ",
            x_y_delimiter="\n ---- \n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Fix this code. ```{question}```",
            targets="Potential fix: ```{answer}```",
            inputs_prefix="input question: ",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="The following code is not correct.\n{question}\n\nCome up with code that fixes this.",
            targets="{answer}",
            inputs_prefix="input question: ",
            targets_prefix="output answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{question}",
            targets="{answer}",
            inputs_prefix="Wrong code: ",
            targets_prefix="Propose solution code: ",
            x_y_delimiter="\n++++++++++\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Broken code:\n{question}",
            targets="{answer}",
            inputs_prefix="Question: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Broken:\n{question}",
            targets="{answer}",
            inputs_prefix="input: ",
            targets_prefix="fixed: ",
            x_y_delimiter="\n****\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="This code is broken:\n{question}\n\nShow the fixed version.",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="Fixed version:\n",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
    ],
    "program_synthesis_dr_repair_error_comments": [
        FewShotPattern(
            inputs="My broken code is below with errors in comments:\n{question}",
            targets="{answer}",
            targets_prefix="The fixed code should be: ",
            x_y_delimiter="\n ---- \n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Examples of incorrect and correct code. {question}",
            targets="{answer}",
            inputs_prefix="Incorrect code: ",
            targets_prefix="Correct code: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Fix this program (see the error comments).\n{question}",
            targets="Ex solution: {answer}",
            inputs_prefix="Ex error code: ",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Come up with a solution for the broken code examples.\n```{question}```",
            targets="{answer}```",
            inputs_prefix="Broken: ",
            targets_prefix="Version which fixes commented errors: ```",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Fix this code. ```{question}```",
            targets="A potential fix:```{answer}```",
            inputs_prefix="QUESTION: ",
            targets_prefix="ANS: ",
            x_y_delimiter="\n",
            example_separator="\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Broken:\n```{question}```",
            targets="{answer}```",
            inputs_prefix="Question: ",
            targets_prefix="Fixed: ```",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Challenging Question. See wrong code:\n{question}",
            targets="{answer}",
            inputs_prefix="input question: ",
            targets_prefix="output answer: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Coding Challenge: fix the errors\n{question}",
            targets="{answer}",
            inputs_prefix="Problem: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Broken code (see error comments):\n{question}",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="Fixed: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="This code is broken:\n{question}",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="Version which fixes commented errors: ",
            x_y_delimiter="\n***\n",
            example_separator="\n\n\n"),
    ],
    "predict_next_turn_dialog_input_inversion": [
        FewShotPattern(
            inputs="Consider this response: {answer}\nWhat was the preceding dialog?",
            targets="{dialog_}",
            targets_prefix="Preceding dialog: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Response: {answer} The preceding conversation:",
            targets="{dialog_}",
            targets_prefix="Preceding conversation: "),
        FewShotPattern(
            inputs="Write an example conversation that led to this. This: {answer}",
            targets="{dialog_}",
            targets_prefix="Preceding conversation: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="See the last examples. Predict the preceding dialog. "
            "{dialog_}",
            targets="{dialog_}",
            targets_prefix="Preceding conversation: ",
            x_y_delimiter="\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="If this is the response, what came before? {answer}",
            targets="{dialog_}",
            inputs_prefix="Problem: ",
            targets_prefix="Before this should be: ",
            x_y_delimiter="\n++++++++++\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Imagine the conversation that came before this response?\n{answer}",
            targets="{dialog_}",
            inputs_prefix="Question:\n",
            targets_prefix="Answer:\n",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="See this dialog response. {answer} What came before?",
            targets="{dialog_}",
            inputs_prefix="Input: ",
            targets_prefix="Came before: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Write the conversation that led to this response. {answer}",
            targets="{dialog_}",
            inputs_prefix="",
            targets_prefix="Conversation:\n",
            x_y_delimiter="\n****\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="What came before. {answer}",
            targets="{dialog_}",
            inputs_prefix="Input: ",
            targets_prefix="Solution: ",
            x_y_delimiter="\n",
            example_separator="\n"),
        FewShotPattern(
            inputs="What might have been said before [{answer}] ???",
            targets="{dialog_}",
            inputs_prefix="Question:\n",
            targets_prefix="Answer:\n",
            x_y_delimiter="\n**********\n",
            example_separator="\n\n\n"),
    ],
    "program_synthesis_dmcc_python_input_inversion": [
        FewShotPattern(
            inputs="If this is the answer: {answer}\n What was the question?",
            targets="{question}",
            targets_prefix="Question: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="This program answers a question. {answer}\nQuestion:",
            targets="{question}",
            targets_prefix="Problem: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="[BEGIN]{answer}[DONE]\n Problem. ",
            targets="{question}",
            inputs_prefix="Code. ",
            targets_prefix="Problem. ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{answer}\n\nProblem this solves:",
            targets="{question}",
            inputs_prefix="Solution: ",
            targets_prefix="Problem: ",
            x_y_delimiter="\n ---- \n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="If this is the answer: {answer}\nWhat's the question?",
            targets="{question}",
            inputs_prefix="[Q]: ",
            targets_prefix="[A]: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="[BEGIN]{answer}[END]",
            targets="{question}",
            inputs_prefix="Solution: ",
            targets_prefix="Problem that it solves: ",
            x_y_delimiter="\n****\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Code solution in Python: {answer}",
            targets="{question}",
            inputs_prefix="",
            targets_prefix="Solution:\n",
            x_y_delimiter="\n++++++++++\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="A: Coding Problem. Solution:\n{answer}",
            targets="Q: {question}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n"),
        FewShotPattern(
            inputs="{answer}",
            targets="{question}",
            inputs_prefix="Code solution: ",
            targets_prefix="Problem it solves: ",
            x_y_delimiter="\n",
            example_separator="\n"),
        FewShotPattern(
            inputs="{answer}",
            targets="{question}",
            inputs_prefix="[solution code]*",
            targets_prefix="The problem: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n"),
    ],
    "program_synthesis_dr_repair_input_inversion": [
        FewShotPattern(
            inputs="Break the fixed code:\n{answer}",
            targets="{question}",
            targets_prefix="Broken Code: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Examples of correct and incorrect code. ```{answer}```",
            targets="{question}```",
            targets_prefix="Incorrect code: ```",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Unfix this program.\n{answer}",
            targets="{question}",
            targets_prefix="Ex wrong program: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Come up with a version of this code that has an error.\n{answer}",
            targets="{question}",
            targets_prefix="Broken: ",
            x_y_delimiter="\n ---- \n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="A potential fix here: ```{answer}```",
            targets="{question}```",
            inputs_prefix="Question: ",
            targets_prefix="Broken version:```",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="The following code is the solution.\n{answer}\n\nPropose an incorrect solution.",
            targets="{question}",
            inputs_prefix="Problem: ",
            targets_prefix="HERE: ",
            x_y_delimiter="\n++++++++++\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Break this code: {answer}",
            targets="{question}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{answer}\nCome up with a wrong code.",
            targets="{question}",
            inputs_prefix="",
            targets_prefix="Wrong code:\n",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{answer}:",
            targets="{question}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{answer}",
            targets="{question}",
            inputs_prefix="Fixed code.\n",
            targets_prefix="Broken code.\n",
            x_y_delimiter="\n",
            example_separator="\n\n"),
    ],
    "natinst_v2": [
        # No Definition given (under-specified) or input/output prefixes.
        FewShotPattern(
            input_pattern="{{Definition}}\n\n{{inputs}}{final_suffix}",
            inputs="{input}",
            targets="{output}",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        # No Definition given (under-specification). Inputs/outputs prefixed.
        FewShotPattern(
            input_pattern="{{Definition}}\n\n{{inputs}}{final_suffix}",
            inputs="{input}",
            targets="{output}",
            inputs_prefix="Ex Input:\n",
            targets_prefix="Ex Output:\n",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        # Definition given. Inputs/outputs prefixed.
        FewShotPattern(
            input_pattern="{{Definition}}\n\n{{inputs}}{final_suffix}",
            inputs="Consider Input: {input}",
            targets="Output: {output}",
            inputs_prefix="Input: ",
            targets_prefix="",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        # Definition given. Inputs/outputs prefixed.
        FewShotPattern(
            input_pattern="{{Definition}}\n\n{{inputs}}{final_suffix}",
            inputs="{input}",
            targets="{output}",
            inputs_prefix="Example Input: ",
            targets_prefix="Example Output: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            input_pattern="{{Definition}}\n\n{{inputs}}{final_suffix}",
            inputs="{input}",
            targets="{output}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n\n",
            example_separator="\n****\n"),
        FewShotPattern(
            input_pattern="{{Definition}}\n\n{{inputs}}{final_suffix}",
            inputs="{input}",
            targets="{output}",
            inputs_prefix="[Q]: ",
            targets_prefix="[A]: ",
            x_y_delimiter="\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            input_pattern="{{Definition}}\n\n{{inputs}}{final_suffix}",
            inputs="{input}",
            targets="{output}",
            inputs_prefix="[EX Q]: ",
            targets_prefix="[EX A]: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            input_pattern="{{Definition}}\n--------\n{{inputs}}{final_suffix}",
            inputs="{input}",
            targets="{output}",
            inputs_prefix="Question: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            input_pattern="TASK DEFINITION: {{Definition}}\n{{inputs}}{final_suffix}",
            inputs="{input}",
            targets="{output}",
            inputs_prefix="PROBLEM: ",
            targets_prefix="SOLUTION: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n"),
        FewShotPattern(
            input_pattern="instruction:\n{{Definition}}\n{{inputs}}{final_suffix}",
            inputs="{input}",
            targets="{output}",
            inputs_prefix="question:\n",
            targets_prefix="answer:\n",
            x_y_delimiter="\n",
            example_separator="\n\n\n"),
    ],
}

FEWSHOT_PATTERNS["wiki_dialog"] = FEWSHOT_PATTERNS["predict_next_turn_dialog"]
FEWSHOT_PATTERNS["task_master"] = FEWSHOT_PATTERNS["predict_next_turn_dialog"]
FEWSHOT_PATTERNS["qrecc"] = FEWSHOT_PATTERNS["predict_next_turn_dialog"]
FEWSHOT_PATTERNS["wiki_dialog_input_inversion"] = FEWSHOT_PATTERNS[
    "predict_next_turn_dialog_input_inversion"]
FEWSHOT_PATTERNS["task_master_input_inversion"] = FEWSHOT_PATTERNS[
    "predict_next_turn_dialog_input_inversion"]
FEWSHOT_PATTERNS["qrecc_input_inversion"] = FEWSHOT_PATTERNS[
    "predict_next_turn_dialog_input_inversion"]

FEWSHOT_PATTERNS["program_synthesis_dr_repair_no_errors"] = FEWSHOT_PATTERNS[
    "program_synthesis_dr_repair"]
FEWSHOT_PATTERNS["program_synthesis_dr_repair_plain_code"] = FEWSHOT_PATTERNS[
    "program_synthesis_dr_repair"]
FEWSHOT_PATTERNS["program_synthesis_dr_repair_line_numbers"] = FEWSHOT_PATTERNS[
    "program_synthesis_dr_repair"]

FEWSHOT_PATTERNS[
    "program_synthesis_dr_repair_no_errors_input_inversion"] = FEWSHOT_PATTERNS[
        "program_synthesis_dr_repair_input_inversion"]
FEWSHOT_PATTERNS[
    "program_synthesis_dr_repair_plain_code_input_inversion"] = FEWSHOT_PATTERNS[
        "program_synthesis_dr_repair_input_inversion"]
FEWSHOT_PATTERNS[
    "program_synthesis_dr_repair_line_numbers_input_inversion"] = FEWSHOT_PATTERNS[
        "program_synthesis_dr_repair_input_inversion"]
FEWSHOT_PATTERNS[
    "program_synthesis_dr_repair_error_comments_input_inversion"] = FEWSHOT_PATTERNS[
        "program_synthesis_dr_repair_input_inversion"]

FEWSHOT_PATTERNS_NO_OPTIONS = copy.deepcopy(FEWSHOT_PATTERNS)
for t_name, templates in FEWSHOT_PATTERNS_NO_OPTIONS.items():
  for ti, few_shot_pattern in enumerate(templates):
    in_template = few_shot_pattern.inputs
    in_template = in_template.replace("\n\n{options_}", "")
    in_template = in_template.replace("\n{options_}", "")
    in_template = in_template.replace(" | {options_}", "")
    in_template = in_template.replace(" {options_}", "")
    in_template = in_template.replace(" ({options_})", "")
    in_template = in_template.replace("{options_}", "")
    in_template = in_template.replace("\n\n{options_str}", "")
    in_template = in_template.replace("\n{options_str}", "")
    few_shot_pattern.inputs = in_template
    FEWSHOT_PATTERNS_NO_OPTIONS[t_name][ti] = few_shot_pattern


INLINE_FS_PATTERNS = {
    "natinst_v2": [
        ("You will be given a definition of a task first, then an example. "
         "Follow the example to solve a new instance of the task.\n"
         "{Definition}\n\n{ex_input}\nSolution: {ex_output}\n"
         "Why? {ex_explanation}\n\n"
         "New input: {input}\nSolution:", "{output}"),
        ("Given the task definition, example input & output, solve the new "
         "input case.\n{Definition}\nExample: {ex_input}\nOutput: "
         "{ex_output}\n{ex_explanation}\n\n"
         "New input case for you: {input}\nOutput:", "{output}"),
        ("Teacher: {Definition}\nTeacher: Now, understand the problem? If "
         "you are still confused, see the following example:\n"
         "{ex_input}\nSolution: {ex_output}\nReason: {ex_explanation}\n\n"
         "Now, solve this instance: {input}\nStudent:", "{output}"),
        ("{Definition}\n\nExample input: {ex_input}\nExample output: "
         "{ex_output}\nExample explanation: {ex_explanation}\nQ: {input}\nA:",
         "{output}"),
        ("Detailed Instructions: {Definition}\nSee one example below:\n"
         "Problem: {ex_input}\nSolution: {ex_output}\n"
         "Explanation: {ex_explanation}\n\n"
         "Problem: {input}\nSolution:", "{output}"),
        ("{Definition}\nExample: {ex_input}\nExample solution: {ex_output}\n"
         "Example explanation: {ex_explanation}\n\nProblem: {input}\n",
         "Solution: {output}"),
        ("{Definition}\nOne example: {ex_input}\nSolution is here: {ex_output}"
         "\nExplanation: {ex_explanation}\n\nNow, solve this: {input}\n"
         "Solution:", "{output}"),
        ("Part 1. Definition\n{Definition}\n"
         "Part 2. Example\n{ex_input}\n"
         "Answer: {ex_output}\n"
         "Explanation: {ex_explanation}\n"
         "Part 3. Exercise\n{input}\nAnswer:", "{output}"),
        ("{Definition}\n\n"
         "Let me give you an example: {ex_input}\n"
         "The answer to this example can be: {ex_output}\n"
         "Here is why: {ex_explanation}\n\n"
         "OK. solve this:\n{input}\nAnswer:", "{output}"),
        ("{Definition}\n"
         "One example is below.\n"
         "Q: {ex_input}\nA: {ex_output}\n"
         "Rationale: {ex_explanation}\n"
         "Q: {input}\nA:", "{output}"),
    ],
}


# All possible option start strings. We will randomly pick one to use, when
# formatting a option string.
OPT_START_STRING_CANDIDATES = [
    "",
    "OPTIONS:",
    "Possible answers:",
    "Available choices:",
    "Options:",
    "OPT:",
    "Choose from:",
    "Choose your answer from:",
    "Available options:",
    "Options are:",
    "Choices:",
    "Pick your answer from:",
    "Select from:",
    "Pick from:",
    "Select from the following.",
    "pick from the following.",
]

ROMAN_NUMS = [
    "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII",
    "XIII", "XIV", "XV", "XVI", "XVII", "XVIII", "XIX", "XX", "XXI", "XXII",
    "XXIII", "XXIV", "XXV", "XXVI"
]
# All possible itemization strings.
# This is an array of char arrays. Each array represents 26 strings, supporting
# up to 26 option items (because there are 26 chars from A to Z). If an example
# has more than 26 options, it will error out.
# We will randomly pick one to use, when formatting a option string.
OPT_ITEM_NAME_CANDIDATES = [
    ["\n - " for _ in range(26)],
    ["\n -" for _ in range(26)],
    ["\n -- " for _ in range(26)],
    ["\n --" for _ in range(26)],
    ["\n + " for _ in range(26)],
    ["\n +" for _ in range(26)],
    ["\n * " for _ in range(26)],
    ["\n *" for _ in range(26)],
    ["\n- " for _ in range(26)],
    ["\n+ " for _ in range(26)],
    ["\n* " for _ in range(26)],
    ["\n[-] " for _ in range(26)],
    ["\n[+] " for _ in range(26)],
    [" - " for _ in range(26)],
    [" -- " for _ in range(26)],
    [" + " for _ in range(26)],
    [" * " for _ in range(26)],
    [" -" for _ in range(26)],
    [" --" for _ in range(26)],
    [" +" for _ in range(26)],
    [" *" for _ in range(26)],
    [" [+] " for _ in range(26)],
    [" [-] " for _ in range(26)],
    [" " + x.lower() + ". " for x in ROMAN_NUMS],  #  i.
    [" [" + x.lower() + "] " for x in ROMAN_NUMS],  #  [i]
    [" (" + x.lower() + ") " for x in ROMAN_NUMS],  #  (i)
    [" (" + x.lower() + "). " for x in ROMAN_NUMS],  #  (i).
    ["\n" + x.lower() + ". " for x in ROMAN_NUMS],  # \ni.
    ["\n[" + x.lower() + "] " for x in ROMAN_NUMS],  # \n[i]
    ["\n(" + x.lower() + ") " for x in ROMAN_NUMS],  # \n(i)
    ["\n(" + x.lower() + "). " for x in ROMAN_NUMS],  # \n(i).
    [" " + x.upper() + ". " for x in ROMAN_NUMS],  #  I.
    [" [" + x.upper() + "] " for x in ROMAN_NUMS],  #  [I]
    [" (" + x.upper() + ") " for x in ROMAN_NUMS],  #  (I)
    [" (" + x.upper() + "). " for x in ROMAN_NUMS],  #  (I).
    ["\n" + x.upper() + ". " for x in ROMAN_NUMS],  # \nI.
    ["\n[" + x.upper() + "] " for x in ROMAN_NUMS],  # \n[I]
    ["\n(" + x.upper() + ") " for x in ROMAN_NUMS],  # \n(I)
    ["\n(" + x.upper() + "). " for x in ROMAN_NUMS],  # \n(I).
    ["\n[" + chr(x) + "]. " for x in range(ord("a"),
                                           ord("z") + 1)],  # \n[A].
    ["\n[" + chr(x) + "]. " for x in range(ord("A"),
                                           ord("Z") + 1)],  # \n[a].
    ["\n[" + str(x) + "]. " for x in range(1, 27)],  # \n[1].
    ["\n(" + chr(x) + "). " for x in range(ord("a"),
                                           ord("z") + 1)],  # \n(A).
    ["\n(" + chr(x) + "). " for x in range(ord("A"),
                                           ord("Z") + 1)],  # \n(a).
    ["\n(" + str(x) + "). " for x in range(1, 27)],  # \n(1).
    ["\n" + chr(x) + "). " for x in range(ord("a"),
                                          ord("z") + 1)],  # \nA).
    ["\n" + chr(x) + "). " for x in range(ord("A"),
                                          ord("Z") + 1)],  # \na).
    ["\n" + str(x) + "). " for x in range(1, 27)],  # \n1).
    ["\n (" + chr(x) + "). " for x in range(ord("a"),
                                            ord("z") + 1)],  # \n (A).
    ["\n (" + chr(x) + "). " for x in range(ord("A"),
                                            ord("Z") + 1)],  # \n (a).
    ["\n (" + str(x) + "). " for x in range(1, 27)],  # \n (1).
    ["\n " + chr(x) + "). " for x in range(ord("a"),
                                           ord("z") + 1)],  # \n A).
    ["\n " + chr(x) + "). " for x in range(ord("A"),
                                           ord("Z") + 1)],  # \n a).
    ["\n " + str(x) + "). " for x in range(1, 27)],  # \n 1).
    [" (" + chr(x) + "). " for x in range(ord("a"),
                                          ord("z") + 1)],  #  (A).
    [" (" + chr(x) + "). " for x in range(ord("A"),
                                          ord("Z") + 1)],  #  (a).
    [" (" + str(x) + "). " for x in range(1, 27)],  #  (1).
    [" " + chr(x) + "). " for x in range(ord("a"),
                                         ord("z") + 1)],  #  A).
    [" " + chr(x) + "). " for x in range(ord("A"),
                                         ord("Z") + 1)],  #  a).
    [" " + str(x) + "). " for x in range(1, 27)],  #  1).
    [" " + chr(x) + ". " for x in range(ord("a"),
                                        ord("z") + 1)],  #  A.
    [" " + chr(x) + ". " for x in range(ord("A"),
                                        ord("Z") + 1)],  #  a.
    [" " + str(x) + ". " for x in range(1, 27)],  #  1.
    ["\n[" + chr(x) + "]. " for x in range(ord("a"),
                                           ord("z") + 1)],  # \n[A]:
    ["\n[" + chr(x) + "]. " for x in range(ord("A"),
                                           ord("Z") + 1)],  # \n[a]:
    ["\n[" + str(x) + "]. " for x in range(1, 27)],  # \n[1]:
    ["\n(" + chr(x) + "). " for x in range(ord("a"),
                                           ord("z") + 1)],  # \n(A):
    ["\n(" + chr(x) + "). " for x in range(ord("A"),
                                           ord("Z") + 1)],  # \n(a):
    ["\n(" + str(x) + "). " for x in range(1, 27)],  # \n(1):
    ["\n" + chr(x) + "). " for x in range(ord("a"),
                                          ord("z") + 1)],  # \nA):
    ["\n" + chr(x) + "). " for x in range(ord("A"),
                                          ord("Z") + 1)],  # \na):
    ["\n" + str(x) + "). " for x in range(1, 27)],  # \n1):
    ["\n (" + chr(x) + "). " for x in range(ord("a"),
                                            ord("z") + 1)],  # \n (A):
    ["\n (" + chr(x) + "). " for x in range(ord("A"),
                                            ord("Z") + 1)],  # \n (a):
    ["\n (" + str(x) + "). " for x in range(1, 27)],  # \n (1):
    ["\n " + chr(x) + "). " for x in range(ord("a"),
                                           ord("z") + 1)],  # \n A):
    ["\n " + chr(x) + "). " for x in range(ord("A"),
                                           ord("Z") + 1)],  # \n a):
    ["\n " + str(x) + "). " for x in range(1, 27)],  # \n 1):
    [" (" + chr(x) + "). " for x in range(ord("a"),
                                          ord("z") + 1)],  #  (A):
    [" (" + chr(x) + "). " for x in range(ord("A"),
                                          ord("Z") + 1)],  #  (a):
    [" (" + str(x) + "). " for x in range(1, 27)],  #  (1):
    [" " + chr(x) + "). " for x in range(ord("a"),
                                         ord("z") + 1)],  #  A):
    [" " + chr(x) + "). " for x in range(ord("A"),
                                         ord("Z") + 1)],  #  a):
    [" " + str(x) + "). " for x in range(1, 27)],  #  1):
    [" " + chr(x) + ". " for x in range(ord("a"),
                                        ord("z") + 1)],  #  A:
    [" " + chr(x) + ". " for x in range(ord("A"),
                                        ord("Z") + 1)],  #  a:
    [" " + str(x) + ". " for x in range(1, 27)],  #  1:
]

# All possible strings to add to the end of all items.
# Currently, we only support up to 26 items.
OPT_ITEM_END_STR_CANDIDATES = [
    ["" for _ in range(26)],
    [";" for _ in range(26)],
    ["." for _ in range(26)],
]

# ----------------------- DIALOG FORMAT -----------------------
# All possible dialog start strings. We will randomly pick one to use in train
DIALOG_START_STRING_CANDIDATES = [
    "",
    "DIALOG:",
    "Dialog:",
    "CONVERSATION:"
    "Conversation:",
    "Convo:",
    "Read the following conversation:",
    "A 2 person conversation:",
    "A 2 person dialog:",
    "A dialog between 2 people:",
    "See the 2 person dialog:",
    "Conversation transcript:",
    "Phone call:",
    "2-way dialog:",
]

# All possible itemization strings for 2 person dialogs.
# We will randomly pick one to use, when formatting a dialog string.
DIALOG_ITEM_NAME_CANDIDATES = [
    ["\n - " for _ in range(1000)],
    ["\n -" for _ in range(1000)],
    ["\n -- " for _ in range(1000)],
    ["\n --" for _ in range(1000)],
    ["\n + " for _ in range(1000)],
    ["\n +" for _ in range(1000)],
    ["\n * " for _ in range(1000)],
    ["\n *" for _ in range(1000)],
    ["\n- " for _ in range(1000)],
    ["\n+ " for _ in range(1000)],
    ["\n* " for _ in range(1000)],
    ["\n[-] " for _ in range(1000)],
    ["\n[+] " for _ in range(1000)],
    [" - " for _ in range(1000)],
    [" -- " for _ in range(1000)],
    [" + " for _ in range(1000)],
    [" * " for _ in range(1000)],
    [" -" for _ in range(1000)],
    [" --" for _ in range(1000)],
    [" +" for _ in range(1000)],
    [" *" for _ in range(1000)],
    [" [+] " for _ in range(1000)],
    [" [-] " for _ in range(1000)],
    ["\n  A. " if x % 2 == 0 else "\n  B. " for x in range(1000)],  # \n  A.
    ["\nA. " if x % 2 == 0 else "\nB. " for x in range(1000)],  # \nA.
    ["\n[A]. " if x % 2 == 0 else "\n[B]. " for x in range(1000)],  # \n[A].
    ["\n[1]. " if x % 2 == 0 else "\n[2]. " for x in range(1000)],  # \n[1].
    ["\n[a]. " if x % 2 == 0 else "\n[b]. " for x in range(1000)],  # \n[a].
    ["\n(A) " if x % 2 == 0 else "\n(B) " for x in range(1000)],  # \n(A)
    ["\n(1) " if x % 2 == 0 else "\n(2) " for x in range(1000)],  # \n(1)
    ["\n[a]. " if x % 2 == 0 else "\n[b]. " for x in range(1000)],  # \n[a].
    ["\n[x]. " if x % 2 == 0 else "\n[y]. " for x in range(1000)],  # \n[x].
    ["\nSpeaker 1: " if x % 2 == 0 else "\nSpeaker 2: " for x in range(1000)],
    ["\nSpeaker A: " if x % 2 == 0 else "\nSpeaker B: " for x in range(1000)],
    ["\nPerson 1: " if x % 2 == 0 else "\nPerson 2: " for x in range(1000)],
    ["\nPerson A: " if x % 2 == 0 else "\nPerson B: " for x in range(1000)],
    ["\nSpeaker 1) " if x % 2 == 0 else "\nSpeaker 2) " for x in range(1000)],
    ["\nSpeaker A) " if x % 2 == 0 else "\nSpeaker B) " for x in range(1000)],
    ["\nPerson 1) " if x % 2 == 0 else "\nPerson 2) " for x in range(1000)],
    ["\nPerson A) " if x % 2 == 0 else "\nPerson B) " for x in range(1000)],
    [
        "\nAnonymous 1) " if x % 2 == 0 else "\nAnonymous 2) "
        for x in range(1000)
    ],
    [
        "\nAnonymous A) " if x % 2 == 0 else "\nAnonymous B) "
        for x in range(1000)
    ],
    ["\n  Person 1) " if x % 2 == 0 else "\n  Person 2) " for x in range(1000)],
    ["\n Person A) " if x % 2 == 0 else "\n Person B) " for x in range(1000)],
    [
        "\n Anonymous 1) " if x % 2 == 0 else "\n Anonymous 2) "
        for x in range(1000)
    ],
    [
        "\n  Anonymous A) " if x % 2 == 0 else "\n  Anonymous B) "
        for x in range(1000)
    ],
    ["\nPerson X: " if x % 2 == 0 else "\nPerson Y: " for x in range(1000)],
    ["\nSpeaker X) " if x % 2 == 0 else "\nSpeaker Y) " for x in range(1000)],
    ["\nP1: " if x % 2 == 0 else "\nP2: " for x in range(1000)],
    ["\n P1) " if x % 2 == 0 else "\n P2) " for x in range(1000)],
]

# All possible strings to add to the end of all items.
# Currently, we only support up to 1000 items.
DIALOG_ITEM_END_STR_CANDIDATES = [
    ["" for _ in range(1000)],
    [";" for _ in range(1000)],
    ["." for _ in range(1000)],
]
