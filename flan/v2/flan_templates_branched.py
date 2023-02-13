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

"""Templates of the original FLAN."""

# pylint: disable=implicit-str-concat

PATTERNS = {
    "rte": [
        ("{premise}\n\nBased on the paragraph above can we conclude that "
         "\"{hypothesis}\"?\n\n{options_}", "{answer}"),
        ("{premise}\n\nBased on that paragraph can we conclude that this "
         "sentence is true?\n{hypothesis}\n\n{options_}", "{answer}"),
        ("{premise}\n\nCan we draw the following "
         "conclusion?\n{hypothesis}\n\n{options_}", "{answer}"),
        ("{premise}\nDoes this next sentence follow, given the preceding "
         "text?\n{hypothesis}\n\n{options_}", "{answer}"),
        ("{premise}\nCan we infer the following?\n{hypothesis}\n\n{options_}",
         "{answer}"),
        ("Read the following paragraph and determine if the hypothesis is "
         "true:\n\n{premise}\n\nHypothesis: {hypothesis}\n\n{options_}",
         "{answer}"),
        ("Read the text and determine if the sentence is "
         "true:\n\n{premise}\n\nSentence: {hypothesis}\n\n{options_}",
         "{answer}"),
        ("Can we draw the following hypothesis from the context? "
         "\n\nContext:\n\n{premise}\n\nHypothesis: {hypothesis}\n\n{options_}",
         "{answer}"),
        ("Determine if the sentence is true based on the text "
         "below:\n{hypothesis}\n\n{premise}\n{options_}", "{answer}"),
        ("Generate a context and a hypothesis.",
         "Context: {premise}\n\nHypothesis: {hypothesis}"),
    ],
    "wsc": [
        ("{context}\n\nAre \"{text1}\" and \"{text2}\" the same "
         "entity?\n\n{options_}", "{answer}"),
        ("{context}\n\nDo \"{text1}\" and \"{text2}\" have the same "
         "meaning?\n\n{options_}", "{answer}"),
        ("Given the following context\n\n{context}\n\nAre \"{text1}\" and "
         "\"{text2}\" the same?\n\n{options_}", "{answer}"),
        ("{context}\n\nDo \"{text2}\" and \"{text1}\" mean the same "
         "thing?\n\n{options_}", "{answer}"),
        ("{context}\n\nAre \"{text2}\" and \"{text1}\" the same thing in the "
         "aforementioned sentence?\n\n{options_}", "{answer}"),
        ("Context:{context}\n\nIs \"{text2}\" the same as "
         "\"{text1}\"?\n\n{options_}", "{answer}"),
        ("Consider this sentence: {context}\n\nAre \"{text2}\" and \"{text1}\""
         " the same?\n\n{options_}", "{answer}"),
        ("Are \"{text1}\" and \"{text2}\" the same in this "
         "sentence?\n{context}\n\n{options_}", "{answer}"),
        ("Is \"{text1}\" the same as \"{text2}\" in this "
         "sentence?\n{context}\n\n{options_}", "{answer}"),
        ("Do \"{text1}\" and \"{text2}\" point to the same thing in the "
         "following sentence?\n\n{context}\n\n{options_}", "{answer}"),
    ],
    "wsc273": [
        ("{context}\n{options_}", "{answer}"),
        ("Complete the passage.\n\n{context}\n{options_}", "{answer}"),
        ("How does this following sentence end?\n\n{context}\n{options_}",
         "{answer}"),
        ("What is the most logical completion for the following "
         "text?\n\n{context}\n{options_}", "{answer}"),
        ("How does this text end?\n\n{context}\n{options_}", "{answer}"),
        ("What happens next?\n\n{context}\n{options_}", "{answer}"),
        ("Complete the following sentence.\n\n{context}\n{options_}",
         "{answer}"),
        ("Fill in the remainder of the sentence.\n\n{context}\n{options_}",
         "{answer}"),
        ("What is the next event?\n\n{context}\n{options_}", "{answer}"),
        ("Complete the rest of the sentence.\n\n{context}\n{options_}",
         "{answer}"),
    ],
    "wic": [
        ("{sentence1}\n{sentence2}\nDoes the word \"{word}\" mean the same "
         "thing in the above two sentences?\n{options_}", "{answer}"),
        ("Sentence 1: {sentence1}\nSentence 2: {sentence2}\nDoes {word} mean "
         "the same thing in these two sentences?\n{options_}", "{answer}"),
        ("Here is one sentence: {sentence1}\nHere is another sentence: "
         "{sentence2}\nDoes the term {word} mean the same thing in both these "
         "sentences?\n{options_}", "{answer}"),
        ("In these two sentences (1) {sentence1} (2) {sentence2}, does the "
         "word {word} mean the same thing?\n{options_}", "{answer}"),
        ("Does word \"{word}\" have the same meaning in the following two "
         "sentences?\n\n{sentence1}\n\n{sentence2}\n\n{options_}", "{answer}"),
        ("Is the word \"{word}\" used in the same way in the following two "
         "sentences?\n\n{sentence1}\n\n{sentence2}\n\n{options_}", "{answer}"),
        ("Does the word \"{word}\" have the same definition in the next two "
         "sentences?\n\n{sentence1}\n\n{sentence2}\n\n{options_}", "{answer}"),
        ("Is {word} used to mean the same thing in the next two "
         "sentences?\n\n{sentence1}\n\n{sentence2}\n\n{options_}", "{answer}"),
        ("Does \"{word}\" mean the same thing in these two "
         "sentences?\n\n{sentence1}\n\n{sentence2}\n\n{options_}", "{answer}"),
        ("Does the word \"{word}\" mean the same thing in \"{sentence1}\" and "
         "\"{sentence2}\"?\n{options_}", "{answer}"),
    ],
    "record": [
        ("Complete the passage.\n\n{passage}\n\n{query}\n\n{options_str}",
         "{answer}"),
        ("{passage}\n\n{query}\n\n{options_str}", "{answer}"),
        ("Find the right ending to this "
         "passage.\n\n{passage}\n\n{query}\n\n{options_str}", "{answer}"),
        ("What's the most logical way to complete this "
         "passage?\n\n{passage}\n\n{query}\n\n{options_str}", "{answer}"),
        ("Write the next sentence.\n\n{passage}\n\n{query}\n\n{options_str}",
         "{answer}"),
        ("How does this story end?\n\n{passage}\n\n{query}\n\n{options_str}",
         "{answer}"),
        ("Write the last sentence in this "
         "story.\n\n{passage}\n\n{query}\n\n{options_str}", "{answer}"),
        ("Compose the next sentence for this "
         "paragraph.\n\n{passage}\n\n{query}\n\n{options_str}", "{answer}"),
        ("What is the most logical completion of this news "
         "story?.\n\n{passage}\n\n{query}\n\n{options_str}", "{answer}"),
        ("How does the sentence end?\n\n{passage}\n\n{query}\n\n{options_str}",
         "{answer}"),
    ],
    "natural_questions": [
        ("Question: {question}?\nAnswer:", "{answer}"),
        ("{question}?", "{answer}"),
        ("Answer the following question:\n\n{question}", "{answer}"),
        ("Answer this question:\n\n{question}?", "{answer}"),
        ("Please answer this question: {question}", "{answer}"),
        ("Answer the question...{question}?", "{answer}"),
        ("What is the answer to this question? {question}", "{answer}"),
        ("Can you tell me the answer to {question}?", "{answer}"),
        ("Next question: {question}", "{answer}"),
        ("Q: {question} A:", "{answer}"),
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
        ("What is the subject line for this email?\n\n{body}", "{subject}"),
        ("Write a subject line for this message:\n\n{body}", "{subject}"),
        ("{body}\nWrite a subject line for this email.", "{subject}"),
        ("Here is an email: {body}\nWhat is a potential subject line for this "
         "email?", "{subject}"),
        ("{body}\nPropose a subject line for this email?", "{subject}"),
        ("This is the content of an email: {body}\nWhat was the subject line "
         "for this email?", "{subject}"),
        ("This is an email\n{body}\n\nWhat is the subject of this email?",
         "{subject}"),
        ("{body}\n\nGenerate a subject line for this email.", "{subject}"),
        ("Write an email with the following subject:\n\n{subject}", "{body}"),
        ("Write an email with the subject line \"{subject}\".", "{body}"),
    ],
    "cnn_dailymail": [
        ("Write highlights for this article:\n\n{text}", "{highlights}"),
        ("Write some highlights for the following article:\n\n{text}",
         "{highlights}"),
        ("{text}\n\nWrite highlights for this article.", "{highlights}"),
        ("{text}\n\nWhat are highlight points for this article?",
         "{highlights}"),
        ("{text}\nSummarize the highlights of this article.", "{highlights}"),
        ("{text}\nWhat are the important parts of this article?",
         "{highlights}"),
        ("{text}\nHere is a summary of the highlights for this article:",
         "{highlights}"),
        ("Write an article using the following points:\n\n{highlights}",
         "{text}"),
        ("Use the following highlights to write an article:\n\n{highlights}",
         "{text}"),
        ("{highlights}\n\nWrite an article based on these highlights.",
         "{text}"),
    ],
    "gigaword": [
        ("Write a short summary for this text: {text}", "{summary}"),
        ("Briefly summarize this sentence: {text}", "{summary}"),
        ("Generate a short summary this sentence:\n{text}", "{summary}"),
        ("What is a shorter version of this:\n\n{text}", "{summary}"),
        ("{text}\n\nWrite a brief summary in a sentence or less", "{summary}"),
        ("{text}\n\nWhat is a very short summary of the above text?",
         "{summary}"),
        ("{text}\nSummarize the aforementioned text in a single phrase.",
         "{summary}"),
        ("{text}\nCan you generate a short summary of the above paragraph?",
         "{summary}"),
        ("Write a sentence based on this summary: {summary}", "{text}"),
        ("Write a sentence based on \"{summary}\"", "{text}"),
    ],
    "multi_news": [
        ("Summarize this article:\n\n{text}", "{summary}"),
        ("Write a summary based on this article:\n\n{text}", "{summary}"),
        ("Article:\n\n{text}\nWhat is a summary?", "{summary}"),
        ("{text}\nWhat is a one-paragraph summary of the above article?",
         "{summary}"),
        ("Here is a news article: {text}\nA summary of this is?", "{summary}"),
        ("News article:\n\n{text}\nWhat is a shorter version of the above "
         "article?", "{summary}"),
        ("{text}\n\nWrite a summary.", "{summary}"),
        ("Article:\n{text}Summary:\n", "{summary}"),
        ("Write an article based on this summary:\n\n{summary}", "{text}"),
        ("{summary}\n\nExpand this summary.", "{text}"),
    ],
    "newsroom": [
        ("{title}\n\n{text}\n\nWrite a one or two sentence summary.",
         "{summary}"),
        ("Please write a short summary for the following "
         "article:\n\n{title}\n\n{text}", "{summary}"),
        ("Please briefly summarize this news article:\n\n{title}\n\n{text}",
         "{summary}"),
        ("{title}\n{text}\nWhat was this article about?", "{summary}"),
        ("{title}\n{text}\nWhat is a short summary of the above article?",
         "{summary}"),
        ("{title}\n\n{text}\nWhat are the most important parts of this text?",
         "{summary}"),
        ("News article: {title}\n\n{text}\nWhat are the most important parts "
         "of this news article?", "{summary}"),
        ("Write an article with the title: \"{title}\"", "{text}"),
        ("Write a title for this article:\n\n{text}", "{title}"),
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
        ("Write a dialog about anything you want", "{dialogue}"),
        ("Write a dialog based on this summary:\n{summary}.", "{dialogue}"),
        ("Write a dialog with this premise \"{summary}\".", "{dialogue}"),
    ],
    "xsum": [
        ("Summarize:\n\n{text}", "{summary}"),
        ("Summarize this article:\n\n{text}", "{summary}"),
        ("Summarize this article in one sentence.\n\n{text}", "{summary}"),
        ("{text}\nWhat is a summary of this text?", "{summary}"),
        ("{text}\nWhat was that article about?", "{summary}"),
        ("{text}\n\nThis article was about:", "{summary}"),
        ("Article:{text}\n\nA summary of the above article is?", "{summary}"),
        ("Article:{text}\n\nSummarize the main points of that article.",
         "{summary}"),
        ("Write an article based on this summary:\n\n{summary}", "{text}"),
        ("Write an article based on this \"{summary}\"", "{text}"),
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
        ("What is the title of this article:\n\n{context}", "{title}"),
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
        ("Write a question about the following article: {context}",
         "{question}"),
        ("{context}\n\nAsk a question about this article.", "{question}"),
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
        ("\n\n{background}\n\n{context}Answer this question using a quote from"
         " the following article:\n\n{question}", "{answer}"),
        ("Which entity is this text about?\n\n{background}\n\n{context}",
         "{title}"),
        ("{background}\n\n{context}\n\nAsk a question about this article.",
         "{question}"),
    ],
    "para_crawl": [
        ("How do you say \"{sent1}\" in {lang2}?", "{sent2}"),
        ("{sent2} How do you say this sentence in {lang1}?", "{sent1}"),
        ("{sent1} Say this using {lang2}", "{sent2}"),
        ("Translate from {lang1} to {lang2}:\n\n{sent1}", "{sent2}"),
        ("Translate from {lang2} to {lang1}:\n\n{sent2}", "{sent1}"),
        ("Translate \"{sent2}\" from {lang2} to {lang1}.", "{sent1}"),
        ("Translate \"{sent1}\" to {lang2}.", "{sent2}"),
        ("Translate the following.\n\n{lang1}: {sent1}\n\n{lang2}:", "{sent2}"),
        ("Write a sentence in {lang1}.", "{sent1}"),
        ("Write a sentence in {lang2}.", "{sent2}"),
    ],
    "wmt16_translate": [
        ("{sent1}\n\nTranslate to {lang2}", "{sent2}"),
        ("{sent2}\n\nTranslate to {lang1}", "{sent1}"),
        ("{sent2}\n\nCould you please translate this to {lang1}?", "{sent1}"),
        ("{sent2}\n\nTranslate this to {lang1}?", "{sent1}"),
        ("Translate to {lang2}:\n\n{sent1}", "{sent2}"),
        ("Translate the following sentence to {lang2}:\n{sent1}", "{sent2}"),
        ("How is \"{sent1}\" said in {lang2}?", "{sent2}"),
        ("Translate \"{sent1}\" to {lang2}?", "{sent2}"),
        ("Write a sentence not in {lang1}.", "{sent2}"),
        ("{sent2}\n\nWhich language is this?", "{lang2}"),
    ],
    "wmt14_enfr": [
        ("{sent1}\n\nTranslate to {lang2}", "{sent2}"),
        ("{sent2}\n\nTranslate to {lang1}", "{sent1}"),
        ("{sent2}\n\nCould you please translate this to {lang1}?", "{sent1}"),
        ("{sent2}\n\nTranslate this to {lang1}?", "{sent1}"),
        ("Translate to {lang2}:\n\n{sent1}", "{sent2}"),
        ("Translate the following sentence to {lang2}:\n{sent1}", "{sent2}"),
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
        ("{lower}\n\nMake this proper case", "{answer}"),
        ("Please capitalize where necessary: {lower}", "{answer}"),
        ("{answer}\n\nMake this lower case", "{lower}"),
    ],
    "fix_punct": [
        ("{no_punct}\n\nAdd punctuation", "{answer}"),
        ("{no_punct}\n\nCan you repeat this sentence, but add in punctuation?",
         "{answer}"),
        ("{no_punct}\n\nWhat is the correctly punctuated version of this "
         "sentence?", "{answer}"),
        ("{no_punct}\n\nPlease fix the punctuation.", "{answer}"),
        ("{no_punct}\n\nCould you correct the punctuation please?", "{answer}"),
        ("Please add punctuation to this: {no_punct}", "{answer}"),
        ("Add punctuation: {no_punct}", "{answer}"),
        ("Add punctuation to the following sentence: {no_punct}", "{answer}"),
        ("Generate a correctly punctuated version of the following text: "
         "{no_punct}", "{answer}"),
        ("What is the version of the following sentence with correct "
         "punctuation?\n\n{no_punct}", "{answer}"),
    ],
    "word_segment": [
        ("{no_space}\nGenerate a sentence using the above characters:",
         "{answer}"),
        ("{no_space}\nWhat's a sentence that uses these characters?",
         "{answer}"),
        ("{no_space}\n\nPlease segment the words.", "{answer}"),
        ("Add spaces: {no_space}", "{answer}"),
        ("Please add spaces between words: {no_space}", "{answer}"),
        ("This text is missing some spaces, please add them: {no_space}",
         "{answer}"),
        ("Add spaces between the words in the following text: {no_space}",
         "{answer}"),
        ("Write the following list of characters into a correctly formed "
         "sentence: {no_space}", "{answer}"),
        ("{answer}\n\nPlease remove spaces between words.", "{no_space}"),
        ("Remove the spaces from the following sentence: {answer}",
         "{no_space}"),
    ],
    "cosmos_qa": [
        ("{context}\n\nQuestion: {question}\n{options_}", "{answer}"),
        ("{context}\n\n{question}\n{options_}", "{answer}"),
        ("{context}\n\nAnswer the following question: {question}\n{options_}",
         "{answer}"),
        ("{context}\n\nBased on the preceding passage, answer the following "
         "question {question}\n{options_}", "{answer}"),
        ("{context}\n\nGive answer the following question using evidence from "
         "the above passage: {question}\n{options_}", "{answer}"),
        ("Context:{context}\nQuestion {question}\nAnswer:\n{options_}",
         "{answer}"),
        ("Read the following article and answer the "
         "question.\n\n{context}\n\n{question}\n{options_}", "{answer}"),
        ("Answer the question about "
         "text:\n\n{context}\n\n{question}\n{options_}", "{answer}"),
        ("Write a question about the article\n\n{context}", "{question}"),
        ("{context}\n\nGenerate a question about the above context.",
         "{question}"),
    ],
    "ag_news_subset": [
        ("{title}\n\n{text}\n\nWhat is this text about?\n{options_}",
         "{answer}"),
        ("{title}\n\n{text}\n\nWhich topic is this article about?\n{options_}",
         "{answer}"),
        ("{text}\nWhich is the best summary of this article?\n{options_}",
         "{answer}"),
        ("{text}\nWhat is this text about?\n{options_}", "{answer}"),
        ("{text}\n\nWhat best summarizes the content of the above "
         "article?\n{options_}", "{answer}"),
        ("Which is this about?\n\n{text}\n\n{options_}", "{answer}"),
        ("Which is an appropriate title for this "
         "article?\n\n{text}\n\n{options_}", "{answer}"),
        ("Select the topic that this about:\n\n{text}\n\n{options_}",
         "{answer}"),
        ("Write a title:\n{text}", "{title}"),
        ("{text}\n\nWhat is a good title for this?", "{title}"),
    ],
    "bool_q": [
        ("{text}\n\nCan we conclude that {question}?\n\n{options_}",
         "{answer}"),
        ("{text}\n\nIs it true that {question}?\n\n{options_}", "{answer}"),
        ("{text}\n\n{question}?\n\n{options_}", "{answer}"),
        ("Text: {text}\n\nQuestion: {question}?\n\n{options_}", "{answer}"),
        ("{text}\n\nWhat's the best answer to this question: "
         "{question}?\n\n{options_}", "{answer}"),
        ("{text}\nBased on the above text, what's the best answer to this "
         "question: {question}?\n\n{options_}", "{answer}"),
        ("{text}\nAnswer this question, making sure that the answer is "
         "supposed by the text: {question}?\n\n{options_}", "{answer}"),
        ("{text}\n\nIs the following statement correct based on the "
         "text\n\n{question}\n\n{options_}", "{answer}"),
        ("{title}\n\n{text}\n\nIs this statement correct "
         "\"{question}\"?\n\n{options_}", "{answer}"),
        ("Is it true that {question} based on the following "
         "text?\n\n{text}\n\n{options_}", "{answer}"),
    ],
    "definite_pronoun_resolution": [
        ("{sentence}\n\nWho is {pronoun} referring to?\n{options_}",
         "{answer}"),
        ("{sentence}\n\nWho is \"{pronoun}\" in this prior "
         "sentence?\n{options_}", "{answer}"),
        ("{sentence}\n\nWho is {pronoun} referring to in this "
         "sentence?\n{options_}", "{answer}"),
        ("{sentence}\nTell me who {pronoun} is.\n{options_}", "{answer}"),
        ("{sentence}\nBased on this sentence, who is {pronoun}?\n\n{options_}",
         "{answer}"),
        ("Who is {pronoun} in the following "
         "sentence?\n\n{sentence}\n\n{options_}", "{answer}"),
        ("Which entity is {pronoun} this sentence?\n\n{sentence}\n\n{options_}",
         "{answer}"),
        ("Who is {pronoun} referring to in the following "
         "sentence?\n{sentence}\n\n{options_}", "{answer}"),
        ("Which person is {pronoun} referring to in the following "
         "sentence?\n{sentence}\n\n{options_}", "{answer}"),
        ("{sentence}\nWho is \"{pronoun}\"?\n{options_}", "{answer}"),
    ],
    "glue_mrpc": [
        ("Here are two sentences:\n{sentence1}\n{sentence2}\nDo they have the "
         "same meaning?\n{options_}", "{answer}"),
        ("Here are two sentences:\n\n{sentence1}\n\n{sentence2}\nAre the two "
         "sentences saying the same thing?\n{options_}", "{answer}"),
        ("{sentence1}\n\n{sentence2}\n\nDo the above sentences mean the same "
         "thing?\n{options_}", "{answer}"),
        ("{sentence1}\n\n{sentence2}\n\nPlease tell me if the sentences above "
         "mean the same.\n{options_}", "{answer}"),
        ("{sentence1}\n{sentence2}\nAre these sentences conveying the same "
         "meaning?\n{options_}", "{answer}"),
        ("{sentence1}\n{sentence2}\nIf the first sentence is true, is the "
         "second one also true?\n{options_}", "{answer}"),
        ("{sentence1}\n{sentence2}\nAre these two sentences paraphrases of "
         "each other?\n{options_}", "{answer}"),
        ("Do the following two sentences have the same "
         "meaning?\n{sentence1}\n{sentence2}\n\n{options_}", "{answer}"),
        ("Do these two sentences mean the same "
         "thing?\n{sentence1}\n{sentence2}\n\n{options_}", "{answer}"),
        ("Do these sentences have the same "
         "meaning?\n{sentence1}\n{sentence2}\n\n{options_}", "{answer}"),
    ],
    "glue_qqp": [
        ("{question1}\n{question2}\nWould you say that these questions are the"
         " same?\n{options_}", "{answer}"),
        ("{question1}\n{question2}\nDo those questions have the same "
         "meaning?\n{options_}", "{answer}"),
        ("{question1}\n{question2}\n\nAre these two questions inquiring about "
         "the same information?\n{options_}", "{answer}"),
        ("{question1}\n\n{question2}\n\nPlease tell me if those questions are "
         "the same.\n{options_}", "{answer}"),
        ("{question1}\n\n{question2}\n\nAre these two questions paraphrases of"
         " each other?\n{options_}", "{answer}"),
        ("First question: {question1}\nSecond question: {question2}\nAre these"
         " two questions asking the same thing?\n{options_}", "{answer}"),
        ("Question 1: {question1}\nQuestion 2: {question2}\nAre questions 1 "
         "and 2 asking the same thing?\n{options_}", "{answer}"),
        ("Question 1: {question1}\nQuestion 2: {question2}\n\nWould the answer"
         " to these two questions be the same?\n{options_}", "{answer}"),
        ("Are the following two questions the "
         "same?\n{question1}\n{question2}\n\n{options_}", "{answer}"),
        ("Do these questions have the same "
         "meaning?\n{question1}\n{question2}\n\n{options_}", "{answer}"),
    ],
    "imdb_reviews": [
        ("{text}\nWhat is the sentiment of this review?\n{options_}",
         "{answer}"),
        ("{text}\nWould you say this review is positive or "
         "negative?\n{options_}", "{answer}"),
        ("{text}\nHow would you describe the sentiment of this "
         "review?\n{options_}", "{answer}"),
        ("{text}\n\nIs the sentiment of this review positive or "
         "negative?\n{options_}", "{answer}"),
        ("{text}\n\nDid this review think positively or negatively of the "
         "movie?\n{options_}", "{answer}"),
        ("Please tell me the sentiment of the following review: "
         "{text}\n{options_}", "{answer}"),
        ("Determine the sentiment:\n\n{text}\n{options_}", "{answer}"),
        ("Write a {answer} movie review.", "{text}"),
        ("Generate a movie review with {answer} sentiment.", "{text}"),
        ("What's an example of a movie review?", "{text}"),
    ],
    "paws_wiki": [
        ("{sentence1}\n{sentence2}\n\nDo these sentences mean the same "
         "thing?\n{options_}", "{answer}"),
        ("{sentence1}\n{sentence2}\n\nAre these two sentences paraphrases of "
         "each other?\n{options_}", "{answer}"),
        ("1. {sentence1}\n2. {sentence2}\n\nAre these two sentences "
         "paraphrases of each other?\n{options_}", "{answer}"),
        ("(1) {sentence1}\n(2) {sentence2}\n\nDo these two sentences mean the "
         "same thing?\n\n{options_}", "{answer}"),
        ("Sentence 1: {sentence1}\nSentence 2: {sentence2}\n\nDo these two "
         "sentences convey the same information?\n\n{options_}", "{answer}"),
        ("Do these two sentences from wikipedia have the same "
         "meaning?\n{sentence1}\n{sentence2}\n\n{options_}", "{answer}"),
        ("Same meaning?\n{sentence1}\n{sentence2}\n\n{options_}", "{answer}"),
        ("Are these paraphrases?\n{sentence1}\n{sentence2}\n\n{options_}",
         "{answer}"),
        ("Do these mean the same?\n{sentence1}\n{sentence2}\n\n{options_}",
         "{answer}"),
        ("Please check if these have the same meaning. Answer \"yes\" if they "
         "do, otherwise \"no\".\n{sentence1}\n{sentence2}", "{answer}"),
    ],
    "sentiment140": [
        ("{text}\nWhat is the sentiment of this tweet?\n{options_}",
         "{answer}"),
        ("{text}\n\nHow would the sentiment of this tweet be "
         "described?\n{options_}", "{answer}"),
        ("{text}\n\nDescribe the sentiment embodied by this tweet.\n{options_}",
         "{answer}"),
        ("Tweet: {text}\nPredict the sentiment of this tweet.\n{options_}",
         "{answer}"),
        ("What is the sentiment of the following "
         "tweet?\nTweet:{text}\n{options_}", "{answer}"),
        ("How would one describe the sentiment of "
         "this tweet?\n{text}\n{options_}", "{answer}"),
        ("Write a tweet that is {answer}.", "{text}"),
        ("What is an example of a tweet?", "{text}"),
        ("Write a {answer} tweet.", "{text}"),
        ("Generate a tweet that has the following sentiment: {answer}",
         "{text}"),
    ],
    "story_cloze": [
        ("{context}\n\nPredict the next sentence.\n{options_}", "{answer}"),
        ("{context}\n\nWhat is the next sentence?\n{options_}", "{answer}"),
        ("{context}\n\nWhat is a natural next sentence?\n{options_}",
         "{answer}"),
        ("{context}\n\nWrite the next sentence.\n{options_}", "{answer}"),
        ("Context: {context}\n\nNow do a next sentence "
         "writing task.\n{options_}", "{answer}"),
        ("Story: {context}\n\nWhat happened next?\n{options_}", "{answer}"),
        ("Write the next sentence in this story.\n\n{context}\n{options_}",
         "{answer}"),
        ("Continue the following story.\n\n{context}\n{options_}", "{answer}"),
        ("Write a story that ends with this sentence.\n\n{answer}",
         "{context} {answer}"),
        ("Write a plausible story that ends with this sentence?\n\nLast "
         "sentence: {answer}", "{context} {answer}"),
    ],
    "copa": [
        ("{premise} What is the {question}?\n\n{options_}", "{answer}"),
        ("Here is a premise:{premise}\n\nWhat is the {question}?\n\n{options_}",
         "{answer}"),
        ("{premise}\n\nWhat is the {question} of the preceding "
         "sentence?\n\n{options_}", "{answer}"),
        ("{premise}\n\nWhat is a plausible {question}?\n\n{options_}",
         "{answer}"),
        ("Based on the following sentence, what is the "
         "{question}?\n\n{premise}\n\n{options_}", "{answer}"),
        ("{premise}\n\n{question}: \n\n{options_}", "{answer}"),
        ("What is the {question} of the following "
         "sentence?\n\n{premise}\n\n{options_}", "{answer}"),
        ("Answer the following question about this "
         "sentence:\n\n{premise}\n\nWhat is the {question}?\n\n{options_}",
         "{answer}"),
        ("Write a sentence.", "{premise}"),
        ("Write two sentences.", "{answer} {premise}"),
    ],
    "winogrande": [
        ("How does the sentence end?\n\n{context}\n\n{options_}", "{answer}"),
        ("Write the next sentence.\n\n{context}\n\n{options_}", "{answer}"),
        ("Continue the following story.\n\n{context}\n\n{options_}",
         "{answer}"),
        ("Complete the following sentence.\n\n{context}\n\n{options_}",
         "{answer}"),
        ("Continue writing the following text.\n\n{context}\n\n{options_}",
         "{answer}"),
        ("How does the sentence end?\n\n{context}", "{answer}"),
        ("Write the next sentence.\n\n{context}", "{answer}"),
        ("Continue the following story.\n\n{context}", "{answer}"),
        ("Complete the following sentence.\n\n{context}", "{answer}"),
        ("Continue writing the following text.\n\n{context}", "{answer}"),
    ],
    "yelp_polarity_reviews": [
        ("{text}\nIs this review positive or negative?\n{options_}",
         "{answer}"),
        ("{text}\nWhat is the sentiment of this review?\n{options_}",
         "{answer}"),
        ("{text}\nWas this review given positively or negatively?\n{options_}",
         "{answer}"),
        ("{text}\nHow would this review be described in terms of "
         "sentiment?\n{options_}", "{answer}"),
        ("Is the following review positive or negative?\n\n{text}\n\n{options_}",
         "{answer}"),
        ("What is the sentiment of the following review?\n{text}\n\n{options_}",
         "{answer}"),
        ("How might one describe the sentiment of this "
         "review?\n{text}\n\n{options_}", "{answer}"),
        ("Write a {answer} yelp review.", "{text}"),
        ("Generate a {answer} review for a place.", "{text}"),
        ("What would be an example of an {answer} review?", "{text}"),
    ],
    "arc": [
        ("{question}\n\n{options_}", "{answer}"),
        ("Question: {question}\n\nAnswer:\n{options_}", "{answer}"),
        ("Question: {question}\n\nWhat is the correct answer to the question "
         "from the following choices?\n{options_}", "{answer}"),
        ("Q: {question}\nWhat is the correct answer to "
         "this question?\n{options_}", "{answer}"),
        ("What is the answer?\n\n{question}\n\n{options_}", "{answer}"),
        ("Answer the question\n\n{question}\n\n{options_}", "{answer}"),
        ("{question}\n\nPick the answer from these options\n\n{options_}",
         "{answer}"),
        ("Write a question you would see in a school textbook.", "{question}"),
        ("What's an example of a grad-school level question?", "{question}"),
        ("I just took a test in school today. What question was I asked?",
         "{question}"),
    ],
    "anli": [
        ("{context}\n\nBased on the paragraph above can we conclude "
         "that \"{hypothesis}\"?\n\n{options_}", "{answer}"),
        ("{context}\n\nBased on that paragraph can we conclude that this "
         "sentence is true?\n{hypothesis}\n\n{options_}", "{answer}"),
        ("{context}\n\nCan we draw the following "
         "conclusion?\n{hypothesis}\n\n{options_}", "{answer}"),
        ("{context}\nDoes this next sentence follow, given the preceding "
         "text?\n{hypothesis}\n\n{options_}", "{answer}"),
        ("{context}\nCan we infer the following?\n{hypothesis}\n\n{options_}",
         "{answer}"),
        ("Read the following paragraph and determine if the hypothesis is "
         "true:\n\n{context}\n\nHypothesis: {hypothesis}\n\n{options_}",
         "{answer}"),
        ("Read the text and determine if the sentence is "
         "true:\n\n{context}\n\nSentence: {hypothesis}\n\n{options_}",
         "{answer}"),
        ("Can we draw the following hypothesis from the context? "
         "\n\nContext:\n\n{context}\n\nHypothesis: {hypothesis}\n\n{options_}",
         "{answer}"),
        ("Determine if the sentence is true based on the text "
         "below:\n{hypothesis}\n\n{context}\n{options_}", "{answer}"),
        ("Generate a context and a hypothesis.",
         "Context: {context}\n\nHypothesis: {hypothesis}"),
    ],
    "coqa": [
        ("{text}\n\nAnswer the following questions:\n{numbered_questions}",
         "{numbered_answers}"),
        ("Read the text and answer the "
         "questions.\n\n{text}\n\n{numbered_questions}", "{numbered_answers}"),
        ("Answer the questions at the end based on the "
         "text.\n\n{text}\n\n{numbered_questions}", "{numbered_answers}"),
        ("\n\n{text}\n\nAnswer this series of "
         "questions:\n\n{numbered_questions}", "{numbered_answers}"),
        ("\n\n{text}\n\nWhat are the answers to this following set of "
         "questions:\n\n{numbered_questions}", "{numbered_answers}"),
        ("\n\n{text}\n\nNow, provide a numbered list of answers to these "
         "questions:\n\n{numbered_questions}", "{numbered_answers}"),
        ("\n\n{text}\n\n{numbered_questions}", "{numbered_answers}"),
        ("\n\n{text}\n\n{numbered_questions}\n\nProvide a "
         "numbered list of answers.", "{numbered_answers}"),
        ("Make use of the article to answer the "
         "questions.\n\n{text}\n\n{numbered_questions}", "{numbered_answers}"),
        ("{text}\n\nBased on the article and the following list of answers, "
         "write a list of questions.\n\n{numbered_answers}",
         "{numbered_questions}"),
    ],
    "opinion_abstracts_rotten_tomatoes": [
        ("{numbered_reviews}\n\nWrite a one sentence summary of the reviews "
         "above.", "{critic_consensus}"),
        ("{numbered_reviews}\n\nWhat is a brief summary of "
         "the following reviews?", "{critic_consensus}"),
        ("{numbered_reviews}\nBased on these individual reviews, "
         "what is the critic consensus?", "{critic_consensus}"),
        ("{numbered_reviews}\nWhat is the consensus?", "{critic_consensus}"),
        ("Here are some reviews for a movie: {numbered_reviews}\n\nWhat was "
         "the overall consensus about the movie?", "{critic_consensus}"),
        ("Summarize the following movie reviews:\n\n{numbered_reviews}",
         "{critic_consensus}"),
        ("Write a one sentence review of the movie \"{movie}\".",
         "{critic_consensus}"),
        ("Write an ordered list of reviews about \"{movie}\".",
         "{numbered_reviews}"),
        ("The critic consesnsus is: {critic_consensus}. What reviews supported"
         " this critic consensus?", "{numbered_reviews}"),
        ("Which movie is the following review about?\n\n{first_review}",
         "{movie}"),
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
        ("Concepts: {concepts}\n\nWrite a sentence that "
         "includes all these words.", "{target}"),
        ("Keywords: {concepts}\n\nWhat is a sentence that includes all these "
         "keywords?", "{target}"),
        ("Here are some concepts: {concepts}\n\nWhat is a sentence about these"
         " concepts?", "{target}"),
        ("Produce a sentence which mentions all of these concepts: {concepts}",
         "{target}"),
        ("Write a sentence about the following things:\n\n{concepts}",
         "{target}"),
        ("Generate a sentence that includes "
         "all the following words: {concepts}", "{target}"),
        ("What are the keywords in the following sentence:\n\n{target}",
         "{concepts}"),
        ("What are the most important words in the following "
         "sentence:\n\n{target}", "{concepts}"),
        ("Identify the most salient words in this sentence:\n\n{target}",
         "{concepts_newline}"),
        ("Generate a sentence, and then tell me the concepts included in that "
         "sentence.", "Sentence:\n{target}\n\nConcepts:\n{concepts_newline}"),
    ],
    "dart": [
        ("Triple: {tripleset}\n\nWhat is a sentence that describes this triple?",
         "{target}"),
        ("Data: {tripleset}\n\nWhat would a sentence about this data be like?",
         "{target}"),
        ("Generate an approximately fifteen-word sentence that describes all "
         "this data: {tripleset}", "{target}"),
        ("Here is some data: {tripleset}.\n\nWrite a sentence that describes "
         "this data", "{target}"),
        ("This is some data: {tripleset}.\n\nGenerate a detailed description "
         "of this data", "{target}"),
        ("Generate a sentence about this data: {tripleset}", "{target}"),
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
         "Write a sentence that includes the following data about a restaurant",
         "{target}"),
        ("Sentence: {meaning_representation}\n\nCan you represent the content "
         "in this sentence in data form?", "{target}"),
        ("Write a sentence about a restaurant with all the following "
         "attributes: {meaning_representation}", "{target}"),
        ("Write a sentence that is about a restaurant with all the following "
         "properties: {meaning_representation}", "{target}"),
        ("Produce a detailed sentence about a restaurant using the following "
         "words: {meaning_representation}", "{target}"),
        ("Generate a descriptive sentence about a restaurant using the "
         "following words:\n\n{meaning_representation}", "{target}"),
    ],
    "web_nlg_en": [
        ("{input_string}\n\nWhat is sentence that verbalizes this data?",
         "{target}"),
        ("Data: {input_string}\n\nSentence about the following data: ",
         "{target}"),
        ("Here is some data: {input_string}.\n\nWrite a sentence that "
         "describes this data", "{target}"),
        ("This is some data: {input_string}.\n\nGenerate a detailed "
         "description of this data", "{target}"),
        ("Generate a sentence about this data: {input_string}", "{target}"),
        ("Generate a sentence that describes the following data: {input_string}",
         "{target}"),
        ("Produce a long descriptive sentence that uses all these words: "
         "{input_string}", "{target}"),
        ("Generate an approximately fifteen-word sentence that describes all "
         "this data: {input_string}", "{target}"),
        ("Sentence: {target}\n\nWhat data can be extracted from this sentence?",
         "{input_string}"),
        ("Sentence: {target}\n\nWhat structured data could we extract from "
         "this sentence?", "{input_string}"),
    ],
    "wiki_lingua_english_en": [
        ("Summarize:\n\n{source}", "{target}"),
        ("Summarize the following:\n{source}", "{target}"),
        ("Summarize this article:\n\n{source}", "{target}"),
        ("Summarize this article in one sentence.\n{source}", "{target}"),
        ("What is a one-sentence summary of the following article?\n{source}",
         "{target}"),
        ("In one sentence, describe what the following article is "
         "about:\n\n{source}", "{target}"),
        ("Article: {source}\n\nWhat is a summary?", "{target}"),
        ("Article: {source}\nWhat is a summary of what this article is about?",
         "{target}"),
        ("Write an article based on this summary:\n\n{target}", "{source}"),
        ("Write an article based on this \"{target}\"", "{source}"),
    ],
    "multirc": [
        ("{paragraph}\n\nQuestion: \"{question}\"\n\nResponse: "
         "\"{response}\"\n\nDoes the response correctly answer the "
         "question?\n\n{options_}", "{answer}"),
        ("{paragraph}\n\nQuestion: \"{question}\"\n\nResponse: "
         "\"{response}\"\n\nBased on the paragraph, is the response to the "
         "question is factually correct?\n\n{options_}", "{answer}"),
        ("{paragraph}\n\nQuestion: \"{question}\"\n\nAnswer: "
         "\"{response}\"\n\nIs this answer correct?\n\n{options_}", "{answer}"),
        ("Paragraph: {paragraph}\n\nQuestion: \"{question}\"\n\nAnswer: "
         "\"{response}\"\n\nBased on the paragraph, is this answer "
         "correct\n\n{options_}", "{answer}"),
        ("{paragraph}\n\nBased on the paragraph, does the response "
         "\"{response}\" correctly answer the question "
         "\"{question}\"?\n\n{options_}", "{answer}"),
        ("{paragraph}\n\nAccording to the above paragraph, the correct answer "
         "to the question \"{question}\" is \"{response}\"?\n\n{options_}",
         "{answer}"),
        ("{paragraph}\n\nAfter reading the above, is \"{response}\" the "
         "correct answer to the question \"{question}\"?\n\n{options_}",
         "{answer}"),
        ("{paragraph}\n\nQuestion: \"{question}\"\n\nAnswer: "
         "\"{response}\"\n\nIs this answer to the question "
         "correct?\n{options_}", "{answer}"),
        ("{paragraph}\nDo you have any questions?", "{question}"),
        ("{paragraph}\nWhat question would one ask from this paragraph?",
         "{question}"),
    ],
    "cb": [
        ("{premise}\n\nBased on the paragraph above can we conclude that "
         "\"{hypothesis}\"?\n\n{options_}", "{answer}"),
        ("{premise}\n\nBased on that paragraph can we conclude that this "
         "sentence is true?\n{hypothesis}\n\n{options_}", "{answer}"),
        ("{premise}\n\nCan we draw the following "
         "conclusion?\n{hypothesis}\n\n{options_}", "{answer}"),
        ("{premise}\nDoes this next sentence follow, given the preceding "
         "text?\n{hypothesis}\n\n{options_}", "{answer}"),
        ("{premise}\nCan we infer the following?\n{hypothesis}\n\n{options_}",
         "{answer}"),
        ("Read the following paragraph and determine if the hypothesis is "
         "true:\n\n{premise}\n\nHypothesis: {hypothesis}\n\n{options_}",
         "{answer}"),
        ("Read the text and determine if the sentence is "
         "true:\n\n{premise}\n\nSentence: {hypothesis}\n\n{options_}",
         "{answer}"),
        ("Can we draw the following hypothesis from the context? "
         "\n\nContext:\n\n{premise}\n\nHypothesis: {hypothesis}\n\n{options_}",
         "{answer}"),
        ("Determine if the sentence is true based on the text "
         "below:\n{hypothesis}\n\n{premise}\n{options_}", "{answer}"),
        ("Generate a context and a hypothesis.",
         "Context: {premise}\n\nHypothesis: {hypothesis}"),
    ],
    "cola": [
        ("Sentence: \"{sentence}\"\nWould a linguist rate this sentence to be "
         "acceptable linguistically?\n\n{options_}", "{answer}"),
        ("{sentence}\n\nHow would you consider the linguistic integrity of the"
         " preceding sentence?\n{options_}", "{answer}"),
        ("Test sentence: \"{sentence}\"\nIs this test sentence a correct "
         "grammatical English sentence?\n\n{options_}", "{answer}"),
        ("Sentence: \"{sentence}\"\nWould a linguist rate this sentence to be "
         "acceptable linguistically?\n\n{options_}", "{answer}"),
        ("Is the following sentence linguistically "
         "acceptable?\n{sentence}\n{options_}", "{answer}"),
        ("Would the following sentence, by the strictest standards, be "
         "considered correct by a linguist?\n\n{sentence}\n{options_}",
         "{answer}"),
        ("Is the next sentence syntactically and semantically "
         "acceptable?\n\n{sentence}\n{options_}", "{answer}"),
        ("Would a linguist find the following sentence to be a valid English "
         "sentence grammatically?\n\n{sentence}\n{options_}", "{answer}"),
        ("Generate short a sentence that is linguistically {answer}",
         "{sentence}"),
        ("Produce a brief English sentence that would be considered "
         "grammatically {answer}", "{sentence}"),
    ],
    "sst2": [
        ("Review:\n{sentence}\nIs this movie review sentence negative or "
         "positive?\n{options_}", "{answer}"),
        ("Short movie review: {sentence}\nDid the critic thinking positively "
         "or negatively of the movie?\n{options_}", "{answer}"),
        ("Sentence from a movie review: {sentence}\nWas the movie seen "
         "positively or negatively based on the preceding "
         "review?\n\n{options_}", "{answer}"),
        ("\"{sentence}\"\nHow would the sentiment of this sentence be "
         "perceived?\n\n{options_}", "{answer}"),
        ("Is the sentiment of the following sentence positive or "
         "negative?\n{sentence}\n{options_}", "{answer}"),
        ("What is the sentiment of the following movie review "
         "sentence?\n{sentence}\n{options_}", "{answer}"),
        ("Would the following phrase be considered positive or "
         "negative?\n\n{sentence}\n{options_}", "{answer}"),
        ("Does the following review have a positive or negative opinion of the"
         " movie?\n\n{sentence}\n{options_}", "{answer}"),
        ("Write a {answer} movie review.", "{sentence}"),
        ("Generate a short movie review that has {answer} sentiment.",
         "{sentence}"),
    ],
    "mnli": [
        ("Premise: {premise}\n\nHypothesis: {hypothesis}\n\nDoes the premise "
         "entail the hypothesis?\n\n{options_}", "{answer}"),
        ("Premise: {premise}\nHypothesis: {hypothesis}\nIs the hypothesis "
         "entailed by the premise?\n{options_}", "{answer}"),
        ("Here is a premise:\n{premise}\n\nHere is a "
         "hypothesis:\n{hypothesis}\n\nIs it possible to conclude that if the "
         "premise is true, then so is the hypothesis?\n{options_}", "{answer}"),
        ("Sentence 1: {premise}\n\nSentence 2: {hypothesis}\nIs this second "
         "sentence entailed by the first sentence?\n\n{options_}", "{answer}"),
        ("Sentence 1: {premise}\n\nSentence 2: {hypothesis}\n\nIf the first "
         "sentence is true, then is the second sentence true?\n{options_}",
         "{answer}"),
        ("Based on the premise \"{premise}\", can we conclude the hypothesis "
         "\"{hypothesis}\" is true?\n\n{options_}", "{answer}"),
        ("Premise: \"{premise}\" If this premise is true, what does that tell "
         "us about whether it entails the hypothesis "
         "\"{hypothesis}\"?\n\n{options_}", "{answer}"),
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
        ("Does the sentence \"{sentence}\" provide a valid answer to the "
         "question \"{question}\"\n{options_}", "{answer}"),
        ("Is \"{sentence}\" a good answer to the question "
         "\"{question}\"\n{options_}", "{answer}"),
        ("Does \"{sentence}\" correctly answer the question of "
         "{question}\n{options_}", "{answer}"),
        ("Does \"{sentence}\" contain the correct answer to "
         "\"{question}\"\n{options_}", "{answer}"),
        ("Q: {question}\n A: {sentence}\n Does the answer correctly answer the"
         " question\n\n{options_}", "{answer}"),
        ("Question: {question}\nAnswer: {sentence}\n Is the question answered "
         "in a satisfactory fashion?\n\n{options_}", "{answer}"),
        ("Question: {question}\n\nIs {sentence} a good answer to this "
         "question?\n\n{options_}", "{answer}"),
        ("Question: {question}\n\nIs \"{sentence}\" the correct "
         "answer?\n\n{options_}", "{answer}"),
        ("Can you generate a question with a factual answer?", "{question}"),
    ],
    "wnli": [
        ("If \"{sentence1}\", can we conclude that \"{sentence2}\"\n{options_}",
         "{answer}"),
        ("If \"{sentence1}\", does it follow that \"{sentence2}\"\n{options_}",
         "{answer}"),
        ("If \"{sentence1}\", is \"{sentence2}\" correct?\n\n{options_}",
         "{answer}"),
        ("Let's say that \"{sentence1}\"\n\nCan we now say that "
         "\"{sentence2}\"?\n\n{options_}", "{answer}"),
        ("\"{sentence1}\" is a true sentence.\n\nDoes this mean that "
         "\"{sentence2}\"?\n\n{options_}", "{answer}"),
        ("Does \"{sentence2}\" appear to be an accurate statement based on "
         "\"{sentence1}\"?\n\n{options_}", "{answer}"),
        ("Can we conclude that \"{sentence2}\" if the statement "
         "\"{sentence1}\" is true?\n\n{options_}", "{answer}"),
        ("Is it possible to draw the conclusion that \"{sentence2}\" if "
         "\"{sentence1}\"?\n\n{options_}", "{answer}"),
        ("Is \"{sentence2}\" true if \"{sentence1}\"?\n\n{options_}",
         "{answer}"),
        ("Sentence 1: \"{sentence1}\"\n\n Sentence 2: \"{sentence2}\"\n\nIs "
         "sentence 2 true, based on sentence 1?\n\n{options_}", "{answer}"),
    ],
    "snli": [
        ("If \"{premise}\", does this mean that \"{hypothesis}\"?\n\n{options_}",
         "{answer}"),
        ("If \"{premise}\", can we conclude \"{hypothesis}\"?\n\n{options_}",
         "{answer}"),
        ("If \"{premise}\", does it logically follow that "
         "\"{hypothesis}\"?\n\n{options_}", "{answer}"),
        ("Based on the sentence \"{premise}\", is the sentence "
         "\"{hypothesis}\" a true sentence?\n\n{options_}", "{answer}"),
        ("Premise: {premise}\n\nHypothesis: {hypothesis}\n\n.Can we conclude "
         "that the hypothesis is true if the premise is true?\n\n{options_}",
         "{answer}"),
        ("Premise: {premise}\n\nHypothesis: {hypothesis}\n\n.Given the "
         "premise, can we conclude the hypothesis?\n\n{options_}", "{answer}"),
        ("Here is a premise: \"{premise}\"\n\nHere is a hypothesis: "
         "\"{hypothesis}\"\n\n.Does the premise tell us whether the hypothesis"
         " is true?\n\n{options_}", "{answer}"),
        ("Is it possible to conclude that \"{premise}\" if "
         "\"{hypothesis}\"?\n\n{options_}", "{answer}"),
        ("Is the premise \"{premise}\" true if \"{hypothesis}\"?\n\n{options_}",
         "{answer}"),
        ("Write a brief sentence.", "{hypothesis}"),
    ],
    "trec": [
        ("What type of thing is the question \"{text}\" asking "
         "about?\n\n{options_}", "{answer}"),
        ("Is the question \"{text}\" asking about an entity, an abbreviation, "
         "a description, a human, a location, or a numeric "
         "entity?\n\n{options_}", "{answer}"),
        ("Would the answer to the question \"{text}\" be an entity, an "
         "abbreviation, a description, a human, a location, or a numeric "
         "value?\n\n{options_}", "{answer}"),
        ("What kind of thing would the answer to the question \"{text}\" be an"
         " entity, an abbreviation, a description, a human, a location, or a "
         "numeric value?\n\n{options_}", "{answer}"),
        ("What is \"{text}\" asking about?\n\n{options_}", "{answer}"),
        ("From the following options, what is the question \"{text}\" asking "
         "about?\n\n{options_}", "{answer}"),
        ("{text}\n\nWhat kind of thing would answer this "
         "question?\n\n{options_}", "{answer}"),
        ("Here is a question: {text}\n\nWould the answer to this question be "
         "an entity, an abbreviation, a description, a human, a location, or a"
         " numeric value?\n\n{options_}", "{answer}"),
        ("Q: {text}\n\nWhich one of the following options would the answer to "
         "this be?\n\n{options_}", "{answer}"),
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
         "(0=\"no meaning overlap\" and 5=\"means the same thing\"), "
         "how similar are the two sentences?\n\n{options_}", "{answer}"),
        ("How similar are the following two "
         "sentences?\n\n{sentence1}\n{sentence2}\n\nGive the answer on a scale"
         " from 0 - 5, where 0 is \"not similar at all\" and 5 is \"means the "
         "same thing\".\n\n{options_}", "{answer}"),
        ("Do the following sentences say the same "
         "thing?\n\n{sentence1}\n{sentence2}\n\nReturn your answer on a scale "
         "from 0 to 5, where 0 is \"not similar\" and 5 is \"very "
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
        ("Continue writing the next sentence in this "
         "paragraph:\n\n{context}\n\n{options_}", "{answer}"),
        ("Continue writing the next sentence.\n\n{context}\n\n{options_}",
         "{answer}"),
        ("This is a test of commonsense. Complete the next "
         "sentence:\n\n{context}\n\n{options_}", "{answer}"),
        ("Write the next sentence in this paragraph:\n\n{context}\n\n{options_}",
         "{answer}"),
        ("How does the next paragraph end?\n\n{context}\n\n{options_}",
         "{answer}"),
        ("What most naturally follows?\n\n{context}\n\n{options_}", "{answer}"),
        ("What happens next?\n\n{context}\n\n{options_}", "{answer}"),
        ("What is the most logical next event?\n\n{context}\n\n{options_}",
         "{answer}"),
        ("Write the next sentence in the following "
         "story.\n\n{context}\n\n{options_}", "{answer}"),
    ],
    "piqa": [
        ("Here is a goal: {goal}\n\nHow would you accomplish this "
         "goal?\n\n{options_}", "{answer}"),
        ("Here is a goal: {goal}\n\nWhich way makes more sense to accomplish "
         "this goal?\n\n{options_}", "{answer}"),
        ("Goal: {goal}\n\nWhich of the following methods is more reasonable "
         "for accomplishing this goal?\n\n{options_}", "{answer}"),
        ("Objective: {goal}\n\nWhich of the following solutions is more sound "
         "in terms of naive physics reasoning?\n\n{options_}", "{answer}"),
        ("How do you do this: {goal}\n\n{options_}", "{answer}"),
        ("What is the best way to: {goal}\n\n{options_}", "{answer}"),
        ("Which of the following solutions is better for the following "
         "goal:\n{goal}\n\n{options_}", "{answer}"),
        ("How would someone go about accomplishing this "
         "goal?\n{goal}\n\n{options_}", "{answer}"),
        ("What's an example of a task that requires knowledge of physical "
         "objects to perform?", "{goal}"),
        ("What kind of task would test someone's ability to perform physical "
         "reasoning?", "{goal}"),
    ],
    "openbookqa": [
        ("{fact}\n{question}\n\n{options_}", "{answer}"),
        ("Read this fact: \"{fact}\"\n\nNow answer this question: "
         "\"{question}\"\n\n{options_}", "{answer}"),
        ("Given the fact \"{fact}\", what is the answer to the question or "
         "completion \"{question}\"\n\n{options_}", "{answer}"),
        ("Knowing that \"{fact}\", how would one answer "
         "\"{question}\"\n\n{options_}", "{answer}"),
        ("Use evidence from the fact that {fact} to answer this question: "
         "\"{question}\"\n\n{options_}", "{answer}"),
        ("Fact: {fact}\nQuestion: {question}\n\nWhat's the "
         "answer?\n\n{options_}", "{answer}"),
        ("Use this fact to answer the question: "
         "{fact}\n\n{question}\n\n{options_}", "{answer}"),
        ("What sentence would provide a factual answer to this question: "
         "\"{question}\"", "{fact}"),
        ("What is a random fact?", "{fact}"),
        ("Generate a sentence that contains a fact.", "{fact}"),
    ]
}
