## Introduction
We propose a end-to-end system which makes use of a recurrent encoder-decoder
deep neural network to translate speech from the Hindi (Fourth most spoken
language in the world) directly to the text in English(First most spoken language).
We apply a slightly modified sequence-to-sequence with attention architecture that
has previously been used for speech recognition and show that it can be repurposed
for this more complex task. To address the lack of Hindi Audio to English Text,
we create our own dataset using Speech and Text from Hindi media files. We
demonstrate that our developed model successfully outperforms the traditional
cascade of ASR and MT models. Although the model can learn the utterance of
common words, it fails to learn uncommon words and the underlying grammar.
Thus, we also propose methods to mitigate this challenge.

## Motivation

### Why Speech Translation?
Valuable for documentation of low-resource languages.
Helpful in crisis relief; Can help workers to respond to requests made in a foreign language.
Extend access of regional media resources to global crowd( Ex: Youtube Auto-Caption). 

### Why End-to-End Model?
Automatic Speech Recognition +  Machine Translation is expensive in terms of resource.
Source Audio, Source Transcription and Target Translation is required.
Error gets compounded between the cascaded models 

### Why Hindi?
Fourth most spoken language.
Media and Market using hindi as spoken language is growing fast.

## Related Works

### Listen, Attend and Spell ( Chan et. al. 2015)
First, seq-2-seq model with attention for Automatic Speech Recognition.
RNN Encoder ( Listen ) → Attention Weights ( Align ) → RNN Decoder ( Spell).
Input:  Source Language Audio; Output: Source Language Text(Grapheme)

### Listen, Attend and Translate ( Berard et. al. 2016)
Seq-2-seq model with attention for Automatic Speech Translations.
RNN Encoder ( Listen ) → Attention Weights ( Align ) → RNN Decoder ( Translate).
Input:  Synthetic Source Language Audio; Output: Target Language Text(Grapheme).

### Seq-2-Seq Models Can Directly Translate Foreign Speech ( Weiss et. al. 2016)
Seq-2-seq model with attention for Automatic Speech Translations.
Novelty: Real Speech & Use of  ‘Strided Convolution + ConvLSTM’ as Encoder.

**The Major Problem with AST is Paucity of Training Data. Following researchs attempts to address this challenge.***

### Low-Resource Speech-to-Text Translation ( Bansal et. al. 2018)
Relatively small dataset used for model construction (~20 Hours of Speech)
Use of word-level decoding instead of character-level.
 
### End-to-End Automatic Speech Translations Of Audiobooks( Berard et. al. 2018)
Used Augmented LibriSpeech corpus based on 1000+ hours of speech from Audiobooks.

### Tied Multitask Learning for Neural Speech Translation ( Anastasopoulos et. al. 2018)
Pre-training on high-resource speech recognition improves low-resource speech-to-text translation( Bansal et. al. 2019)

## Data and Preprocessing

Speech and translation was extracted from Audio and Subtitles of  20 Hindi Movies. (Collectively 20+Hrs of Audio, 9.5K Unique Words and 150K Occurences ).

### Audio Pre-Processing:
1. Audio was split to clips based on timestamps provided in subtitles file.
2. Clip corresponding to Music was eliminated
3. 80-Dimensional Mel Fbank, Delta, and Delta-Delta Features were extracted for windows of 25ms and Stride of 10ms.  (Decomposition of Audio based on frequency(Hz).)

### Text Pre-Processing:
1. Basic Cleaning: Removed Punctuations and Lowercase each character.
2. Lemmatized words using NLTK package. (Vocab Reduction: 2k Words)
3. Eliminated Words with Less than 10 Occurrences and replaced them with placeholder in the text.( Vocab Reduction: 6k words; Occurence Reduction: ~15k)

### Final Vocab: 1322 Words; Total Occurrences: 135k Words; Audio-Text Pair: 28k; Max Audio Clip Length: 10 Seconds; Max Sentence Length: 17 Words.

## Model Architecture
![model](https://github.com/Shivam0712/End-to-End_Speech-to-Text_Translation/blob/master/Architecture.png)

## Model Training Settings:

1. Train Set: 16.8k(60%) Dev Set: 5.6k(20%) Test Set: 5.6k(20%) 
2. Baseline Model: Google Speech-to-Text(ASR) + Google Translate API(MT)
3. Batch Size: 16 ; Learning Rate: 0.003 ; Epochs: 120
4. Loss Criterion: Cross Entropy Loss
5. Evaluation: Uni-Gram BLEU Score & Sentence BLEU Score

### Please note: 
1. Teacher Forcing Ratio of 0.5 was used
2. To save train time Softmax layer was not used during training

## Results
![model](https://github.com/Shivam0712/End-to-End_Speech-to-Text_Translation/blob/master/Results1.PNG)

![model](https://github.com/Shivam0712/End-to-End_Speech-to-Text_Translation/blob/master/Results2.PNG)
## Conclusions

### Pros:
1. With the help of Encoder-Attention-Decoder structure, End-to-End automatic speech translation model is able successfully outperform traditional ASR+MT model.

### Cons:
2. The uni-gram and sentence BLEU score of the model are still not good enough to be used in applications.
3. Although with sufficient training data, the model is capable of learning utterances of frequent words from audio, it lacks in learning the grammar of sentences.

### Future Work:
1. Accumulate more training data.
2. Use of Beam Decoder to improve performance.
3. Use of word2vec for embedding of target sentences.






