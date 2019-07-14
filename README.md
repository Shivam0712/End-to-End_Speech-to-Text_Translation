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
