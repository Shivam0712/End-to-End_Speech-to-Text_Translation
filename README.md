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




