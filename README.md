# ADVERSARIAL NEURAL CRYPTOGRAPHY

## NOTES 

- Keep Restarting runtime until you get a Turing architecture GPU(Tesla V4) not a K80       

        physical_device_desc: "device: 0, name: Tesla T4, pci bus id: 0000:00:04.0, compute capability: 7.5" 


## UPDATES 

- MIT UTRC Poster -- 7Th AUG 
- AAAI Abstract -- September 18, 2020 (11:59 PM PDT): Electronic Abstracts Due
- Paper ot https://www.nature.com/natmachintell 
















ARCHIVED 
-----------


1. Change in the NN model for Adversaries(s) and Alice/Bob 
    1. filter size,batch size, learning, loss fxn, Key size<< plaintext(round/block cipher)
    2. PRNG @RISHABH C 
    3. Hyprparam tuning Lib @RISHABH C 
2. ~~Experiment single | multiple-adversary~~
    1. ~~Key(L) + Ciphertext → Plaintext(Eve1) + Key(Eve2) PG 21~~
    2. ~~Key(L) + Ciphertext + Plaintext →Plaintext(Eve1) + Key(Eve2)~~

~~Enigma RNN~~ @Avani G ~~~~@RISHABH C 
~~~~~~Key Leaking code~~ @Pranav K ~~~~


3. Security Analysis 
    1. X^2
    2. K-S aproach
    3. NIST Stat.
    4. Fxn Dist XOR
    
4. RNN Viz @Pranav K















https://github.com/HendrikStrobelt/LSTMVis

[https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwiQm6-zmrHrAhVvzDgGHdS2AL0QFjABegQIARAB&url=http%3A%2F%2Fgreydanus.github.io%2F2017%2F01%2F07%2Fenigma-rnn%2F&usg=AOvVaw17jbifsY3bWkdemXNHtYya](https://meet.google.com/linkredirect?authuser=0&dest=https%3A%2F%2Fwww.google.com%2Furl%3Fsa%3Dt%26rct%3Dj%26q%3D%26esrc%3Ds%26source%3Dweb%26cd%3D%26ved%3D2ahUKEwiQm6-zmrHrAhVvzDgGHdS2AL0QFjABegQIARAB%26url%3Dhttp%253A%252F%252Fgreydanus.github.io%252F2017%252F01%252F07%252Fenigma-rnn%252F%26usg%3DAOvVaw17jbifsY3bWkdemXNHtYya)

http://vision.stanford.edu/pdf/KarpathyICLR2016.pdf


5. Transmission Encoder/decoding model  schemes
    1. Different encoding [Kolmogorov complexit](https://en.wikipedia.org/wiki/Kolmogorov_complexity)y and ↔ data capture by adversary

https://www.rapidtables.com/convert/number/ascii-to-binary.html

https://jeremykun.com/2012/04/21/kolmogorov-complexity-a-primer/


https://people.cs.uchicago.edu/~fortnow/papers/kaikoura.pdf
https://www.cs.princeton.edu/courses/archive/fall11/cos597D/L10.pdf

https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0096223




https://github.com/EXYNOS-999/MIT_utrc

https://docs.google.com/document/d/1qA1c5bd0Ydv-99bEp_LjlsccGDdhVUF5OQ86cN8zAY0/edit?usp=sharing


[https://docs.google.com/document/d/1qA1c5bd0Ydv-99bEp_LjlsccGDdhVUF5OQ86cN8zAY0/edit](https://docs.google.com/document/d/1qA1c5bd0Ydv-99bEp_LjlsccGDdhVUF5OQ86cN8zAY0/edit)

cryto RNN https://github.com/avani17101/crypto-rnn
blogpost https://greydanus.github.io/2017/01/07/enigma-rnn/

https://colab.research.google.com/drive/1AbHVW4C5pDGE0M-neF7CfT8HGPFzT5Mt?usp=sharing



----------


## 




6. Paper structure 













RNN  adversarial code 


 neural GPU https://arxiv.org/abs/1511.08228

Multi-agent-RNN-Adversaries modelled on symmetric-key cryptographic attacks(KPA/KCA) to *learn* encryption schemes for communication without knowledge of pre-specified cryptographic algorithms.

//change loss function ( should we do a perfect letter match or should we have a threshold here?)
threshold is good I guess since the encoding makes it more difficult + there is 3 bit noise as well 
Yeah. Maybe we could experiment with different encoding also? 

Yeah sounds good only lowercase maybe that is  hard 5 bit limit /or other encoding 

//also calculate the % intercepted communication as from the loss we don;t get an idea of the strength of the adversary/encryption?

// make key smaller than the text 


    RNN model @Avani G @RISHABH C 
[x] RNN adversary modelling @RISHABH C @Avani G 
[ ] RNN multiple adversary same loss @Avani G
[ ]  secret key leak @Pranav K
[ ] EVE1 EVE2 loss function
[ ] 



             



[ ] Change formulation of loss function @RISHABH C EVE1 EVE 2 

key +plaintext+ciphertext→ eve1(plaintext)+eve2(secretkey)
 

[ ] Encoding/Decoding Model + Noise @RISHABH C 
[ ] Security Analysis  @Pranav K 
    . Nechvatal, M. Smid, E. Barker, A statistical testsuite for random and pseudorandom number generators for cryptographicapplications, Tech. rep., Booz-Allen and Hamilton Inc Mclean Va (2001).

Someone has written a python implementation: https://github.com/dj-on-github/sp800_22_tests

    https://nvlpubs.nist.gov/nistpubs/Legacy/SP/nistspecialpublication800-22r1a.pdf
    https://csrc.nist.gov/projects/cryptographic-algorithm-validation-program   toolkit 


    https://csrc.nist.gov/Projects/Random-Bit-Generation/Documentation-and-Software
    https://csrc.nist.gov/CSRC/media/Projects/Random-Bit-Generation/documents/sts-2_1_2.zip






[ ] Case study: Transmission of information @doc 




----------


[ ] Code and viz. tracking on W&B  @Avani G 


[x] (KPA/CPA) https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5982701/ @Pranav K Let me know if you need any help :)


----------

CODE:


https://github.com/mathybit/ml-tutorials/blob/master/adversarial_neural_cryptography.ipynb













----------





[ ] Writing paper @doc on Overleaf 

Abstract 
Statistical Analysis of the learnt function for encryption of communication


    Change the structure of the information flow.
        Adversary(Eve) uses a different attack.
            1. Chosen Plaintext Attack.
            2. Known Plaintext Attack
            

Sounds good?Feel free to add/delet                     1000100100001000111 
was saying we need to add loss on decoded msg too. Problem with taking loss only on ascii will be our goal being deviated for NN from reconstruction of original msg to jusr reconstrcuting the encoding. Like if 1 bit in asci is predicted wrong still by other bits in original msg (suppose those predicted correct) eve can guess if loss on msg is there, hence loss on decoded msg lena h
loss must be = ascii +1/k decoded msg or similar

aree no tut, we just taking loss on ascii wont make sense, since certain chars can be guess even if wrong in ascii from saorrunding msg by eve



1 2 3 4 5 6 7 8   
   C ↔P= 0101 0001 → C

          P→A(P)→C 
            C→ 

1 2 3 4 5   + 3P



Research Question: 

1. Can Neural Networks come up with cryptographically strong schemes for communication of information in the presence of adversarial agents without knowledge of pre-specified cryptographic algorithms.
2. Multi-adversary neural cryptography
    1. Adversaries having different goal(target/loss fxn)→ Does this improve the network against the specific attacks? https://www.cs.clemson.edu/course/cpsc424/material/Cryptography/Attacks%20on%20Symmetric%20Key.pdf
    2. Can we train Neural Networks to specifically resist a particular adversarial attack?
    3. Change the goal of Alice and Bob; make them “aware” of Eve/multiple Eve…
3. Move beyond simple OTP communication structure?
4. Use Adv. Neural Networks to attach traditional cryptographic ciphers?

TL;DR 
Can Neural Networks “learn” to encrypt data?


2. Interpretebility of the Encryption Scheme that the agents “learn” to communicate.


3. What effect does changing the architecture of the system have on the cryptographic scheme?
    ref: Tuneable parameters:
    Number of parameters/layers of Alice/Bob/Eve. See Also Eve++
    Number of adversaries 
    Encryption and Decryption functions need not be the same.
    Loss function definition
    Make the key smaller ie. trasmitted message longer
    Dynamic Training


## Discussion


# ENCODING/DECODING

Try fixed length encoding first so that the RNN can learn the freq dist. and then increase the fixed length.


# **UCS Encodings(Fixed Length Encoding)**

UCS 2 

## **UCS-2**
    **UCS-2** is **16-bit fixed-width** encoding (***2 bytes***), which means **16 bits** will be used to encode a character. Since a character takes **2 bytes** of memory, **65,536** characters (²¹⁶) characters can be represented by UCS-2 encoding

UCS 4 

    UCS-4 is 32-bit fixed-width encoding (4 bytes), which means **32 bits** will be used to encode a character. Since a character takes **4 bytes** of memory, **4,294,967,296** characters (²³²) characters can be represented by UCS-4


# UTF Encodings (Variable Encodings)


# **UTF-8**

UTF-8 is an **8-bit variable-length** encoding scheme designed to be **compatible** with **ASCII** encoding. In this encoding, from **1** up to **4** bytes can be used to encode a character (*unlike ASCII which uses fixed 1 byte*). UTF-8 encoding uses the **UTF character set** for character code points.
In a **variable-length encoding** scheme, a character is encoded in **multiple** of **N** bits. For example, in UTF-8 encoding, a character can take **M x 8** bits of memory, where **N** is **8** (*fixed*) and **M** can be **1** up to **4** (*variable*).
Here, **N** is also called as the **code unit**. The code unit is the building block of the **code point** (*coded character representation*). In UTF-8 encoding, the code unit is **8 bits** or **1 byte** because a character is encoded in N **bytes**.
The main idea behind UTF-8 was to encode all the characters that could possibly exist on the planet but at the same time support ASCII encoding. This means that an ASCII encoded character will look exactly similar in UTF-8.



NOTE: UTF SUPERSET OF ASCII 


# **UTF-16**

UTF-16 is **16-bit variable length** encoding scheme and it uses the UTF character set for character code points. This means that a UTF-16 encoded character will have a **16-bit code unit**.
As we know that a UTF-8 encoded character can be represented in 1 to 4 code units, a UTF-16 character can be represented in **1 or 2** code units. Hence a UTF-16 character can take **16** or **32** bits of memory based on its **code point**.






# **UTF-32**

UTF-32 is 32-bit **fixed-width** encoding scheme which means a single 32-bit code-unit will be used to encode a character.
In contrast with UTF-8 and UTF-16, since we are not dealing with multiple code units, encoding in UTF-32 is fairly easy. We just need to **convert the code point of a character in a 32-bit binary number**.




Then use variable length encoding more difficult for the RNN 

https://danielmiessler.com/study/encoding-encryption-hashing-obfuscation/


The purpose of *encoding* is to transform data so that it can be properly (and safely) consumed by a different type of system, e.g. binary data being sent over email, or viewing special characters on a web page. The goal is **not** to keep information secret, but rather to ensure that it’s able to be properly consumed.
Encoding transforms data into another format using a scheme *that is publicly available* so that it can easily be reversed. It does not require a key as the only thing required to decode it is the algorithm that was used to encode it.

We are not actually increasing the entropy(in its formal definition) but its [Kolmogorov complexity](https://en.wikipedia.org/wiki/Kolmogorov_complexity)

https://people.cs.uchicago.edu/~fortnow/papers/kaikoura.pdf


- [UTF-1](https://en.wikipedia.org/wiki/UTF-1), a retired predecessor of UTF-8, maximizes compatibility with [ISO 2022](https://en.wikipedia.org/wiki/ISO/IEC_2022), no longer part of *The Unicode Standard*
- [UTF-7](https://en.wikipedia.org/wiki/UTF-7), a 7-bit encoding sometimes used in e-mail, often considered obsolete (not part of *The Unicode Standard*, but only documented as an informational [RFC](https://en.wikipedia.org/wiki/Request_for_Comments), i.e., not on the Internet Standards Track)
- [UTF-8](https://en.wikipedia.org/wiki/UTF-8), uses one to four bytes for each code point, maximizes compatibility with [ASCII](https://en.wikipedia.org/wiki/ASCII)
- [UTF-EBCDIC](https://en.wikipedia.org/wiki/UTF-EBCDIC), similar to UTF-8 but designed for compatibility with [EBCDIC](https://en.wikipedia.org/wiki/EBCDIC) (not part of *The Unicode Standard*)
- [UTF-16](https://en.wikipedia.org/wiki/UTF-16), uses one or two 16-bit code units per code point, cannot encode surrogates
- [UTF-32](https://en.wikipedia.org/wiki/UTF-32), uses one 32-bit code unit per code point

BASE-64 

https://en.wikipedia.org/wiki/Base64#MIME


    **MIME**[[edit](https://en.wikipedia.org/w/index.php?title=Base64&action=edit&section=10)]
    *Main article:* [*MIME*](https://en.wikipedia.org/wiki/MIME)
    The [MIME](https://en.wikipedia.org/wiki/MIME) (Multipurpose Internet Mail Extensions) specification lists Base64 as one of two [binary-to-text encoding](https://en.wikipedia.org/wiki/Binary-to-text_encoding) schemes (the other being [quoted-printable](https://en.wikipedia.org/wiki/Quoted-printable)).[[5]](https://en.wikipedia.org/wiki/Base64#cite_note-rfc_2045-5) MIME's Base64 encoding is based on that of the [RFC 1421](https://tools.ietf.org/html/rfc1421) version of PEM: it uses the same 64-character alphabet and encoding mechanism as PEM, and uses the `=` symbol for output padding in the same way, as described at [RFC 2045](https://tools.ietf.org/html/rfc2045).
    MIME does not specify a fixed length for Base64-encoded lines, but it does specify a maximum line length of 76 characters. Additionally it specifies that any extra-alphabetic characters must be ignored by a compliant decoder, although most implementations use a CR/LF [newline](https://en.wikipedia.org/wiki/Newline) pair to delimit encoded lines.
    Thus, the actual length of MIME-compliant Base64-encoded binary data is usually about 137% of the original data length, though for very short messages the overhead can be much higher due to the overhead of the headers. Very roughly, the final size of Base64-encoded binary data is equal to 1.37 times the original data size + 814 bytes (for headers). 

Other leagcy encoders:
https://encoding.spec.whatwg.org/#security-background

Simple ASCII implementation using padding:

    
    block_size_unpadded = 5
    block_padding = 3
    block_size = block_size_unpadded + block_padding
    
    chrlist = [
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
        'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
        'u', 'v', 'w', 'x', 'y', 'z', '.', ',', '!', '?',
        ':', ' '
    ]
    binlist = [
        '00000', '00001', '00010', '00011', '00100', 
        '00101', '00110', '00111', '01000', '01001',
        '01010', '01011', '01100', '01101', '01110', 
        '01111', '10000', '10001', '10010', '10011',
        '10100', '10101', '10110', '10111', '11000',
        '11001', '11010', '11011', '11100', '11101', 
        '11110', '11111'
    ]
    
    def randombits(n):
        if n == 0:
            return ''
        decvalue = np.random.randint(0, 2**n)
        formatstring = '0' + str(n) + 'b'
        return format(decvalue, formatstring)
    
    def encstr(message, block_padding=0):
        cipher = ""
        for c in message:
            binstr = binlist[chrlist.index(c)]
            binstrpadded = randombits(block_padding) + str(binstr)
            cipher = cipher + binstrpadded
        return cipher, len(message)
    
    def decstr(cipher, n, block_padding=0):
        message = ""
        cipherlength = len(cipher)
        block_size = cipherlength // n
        for i in range(n):
            blockpadded = cipher[block_size*i : block_size*i + block_size]
            blockunpadded = blockpadded[block_padding:]
            character = chrlist[binlist.index(blockunpadded)]
            message = message + character
        return message



CODE IMPLEMENTATIONS:



https://github.com/zanaptak/BinaryToTextEncoding



https://github.com/mmontagna/generic-encoders


SIMPLE ASCII ENCODER


https://github.com/viisual/ASCII-Decorator


~~A real world example with real-world data would be nice, specifically one in which for some reason it's not appropriate to apply a standard cryptography solution.~~

    ~~8-bit ASCII character encodings, and convert them to binary arrays that can be fed into the Alice model for transmission of text.~~
    ~~Additing dimentions to the ciphertext?~~
        ~~https://mathybit.github.io/adversarial-neural-crypto/~~
        ~~*The high level of accuracy by the decryption algorithm makes this a good model to build a toy-model cryptosystem that can operate on human-readable characters. To this end, one could use 8-bit ASCII character encodings, and convert them to binary arrays that can be fed into the Alice model.*~~
        ~~*Instead we limited ourselves to lowercase letters and a few punctuation marks by using a 5-bit binary encoding. We pad this with 3 random bits, resulting in an 8-bit binary encoding that can be fed to the Alice model (this also makes our encryption probabilistic).*~~
        ~~*The fact that our encodings are 32-bit floating point numbers means there’s no nice way to convert the ciphertext to something human-readable. To get around this, we convert each coordinate of the 8-dim encoding into a 32-bit binary string, and concatenate them together to get a binary encoding. This means that each human-readable character goes from an 8-bit padded binary representation to a*~~ 
        ~~*32.8=256*~~
        ~~*-bit ciphertext representation. For example, the plaintext “adi” has (one of many) binary representation*~~

~~>>Extending to multiparty >3 NN for communication with number of keys= number of NN, so that the communication b/w A<>B is secured separately from B<>C and vica versa, problem with extending to multiparty(>3) Neural networks is training time, they suggest having a oracle with the trained parameters which are then given to the neural networks which want to communicate thus reducing time to train.~~

~~>>Also it would be interesting to see what happens in the case of a multi-adversary senario I did not see that case explored in any of the papers.~~


Reviews from ICLR Rejection:
1) As raised in the pre-review, Eve should actually be stronger then Alice and Bob in order to be able to compensate for the missing key. The authors noted they have been doing these experiments and are going to add the results.

Eve++ → 2 FC layers 

3 Party Computation

https://drive.google.com/file/d/1T51Hf3-tjXW97KCysvt7HZOy_7h6LWwy/view?usp=sharing


[https://drive.google.com/file/d/1T51Hf3-tjXW97KCysvt7HZOy_7h6LWwy/edit](https://drive.google.com/file/d/1T51Hf3-tjXW97KCysvt7HZOy_7h6LWwy/edit)

# POTENTIAL CONTRIBUTIONS 


https://docs.google.com/document/d/1qA1c5bd0Ydv-99bEp_LjlsccGDdhVUF5OQ86cN8zAY0/edit?usp=sharing


[https://docs.google.com/document/d/1qA1c5bd0Ydv-99bEp_LjlsccGDdhVUF5OQ86cN8zAY0/edit](https://docs.google.com/document/d/1qA1c5bd0Ydv-99bEp_LjlsccGDdhVUF5OQ86cN8zAY0/edit)



## 


https://www.dropbox.com/s/ksv9vwpj4ks7i36/zhou2019.pdf?dl=0


>> Comments https://www.dropbox.com/s/ksv9vwpj4ks7i36/zhou2019.pdf?dl=0

Multiple adversaries with different goals

# Amazing work!


https://github.com/mathybit/ml-tutorials/blob/master/adversarial_neural_cryptography.ipynb

    
    
    Result Reproduction
    CNN/NN viz and explainability 
    Different Architectures of Alice Bob Eve 
    Eve is a Neural Network only given access to the cipher text.
    Chosen Plaintext Cipher
        [https://youtu.be/ZjYzrn8M3w4](https://youtu.be/ZjYzrn8M3w4)
        [https://youtu.be/EyS1uRvR1RE](https://youtu.be/EyS1uRvR1RE)
    ~~Neural Networks are Turing Complete so ideally can learn any encryption function given enough resources~~
    ~~Use of GANs~~ 
    ~~Generation of plaintext data and corresponding keys~~ 
    ~~Reformulate the goal/target function of Eve to be inplicit(using prior knowledge)~~
    When is the Encryption Scheme secure?
        Original paper if Eve(untrained) cannot converge in (5) epohs. ← Loose definition 
            Change definition 
    When are the messages truly encrypted?
    Increase in the complexity of Eve: https://openreview.net/forum?id=S1HEBe_Jl&noteId=Sku1ifszg


    Chosen Plaintext Attack already discussed here:
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5982701/

    [https://app.lucidchart.com/invitations/accept/e8632f3e-130d-4762-a2d8-b435a5f43335](https://app.lucidchart.com/invitations/accept/e8632f3e-130d-4762-a2d8-b435a5f43335)
    

Neural Cryptography: From Symmetric Encryption to Adversarial Steganography
*Verbatum:*
Instead of having Eve produce the plaintext,P, without the keyK, from the ciphertextC,we think a better construction would be to askEve given a plaintext,Pand a possible ciphertextC′, does C′=Alice(P)? This lends itself toa more natural GAN description where Eve actsas a discriminator. The networks can continue tobe trained adversarially where Eve’s loss is nowbinary cross-entropy on the truth label:

LE:=−∑i(y′ilog(yi) + (1−y′i) log(1−yi))

This architecture, visually, looks as follows:


![](https://paper-attachments.dropbox.com/s_3CFDCB8D3650055E98C5C004EC76B6E92B0FCE039DEA58852291EE13B26AF7BA_1597429567850_Screenshot_2020-08-14+Modesitt-Henry-Coden-Lathe-NeuralCryptography+1+pdf.png)


~~Addition of Noise?~~
Tuneable parameters:

    Number of parameters/layers of Alice/Bob/Eve. See Also Eve++
    Number of adversaries 
    Encryption and Decryption functions need not be the same.
    Loss function definition
    Make the key smaller ie. trasmitted message longer


    
     


----------


Research Domains:

    Explore the encryption scheme that A↔ B that Alice and Bob LEARN.
        How does changing the structure of Eve effect this scheme.
        How does changing the parameters of A/B effect the encryption scheme.


@RISHABH C Change the formulation of the loss function and study the effect it has on the encryption scheme.

        ref:Neural Cryptography 



    
    - 3 party computation 
        - Define Architecture
        
        
    
    
    
    ~~Use of other optimisation methods to find the optimum Adversary~~
        1. ~~Simulated annealing~~
        2. ~~Levenberg-Marquardt Algorithm (LMA)~~




        

 



References:

https://web.archive.org/web/20170614092259/https://asecuritysite.com/encryption/c_c
https://courses.csail.mit.edu/6.857/2018/project/Modesitt-Henry-Coden-Lathe-NeuralCryptography.pdf


METHODS/IMPLEMENTATIONS:

**Implementation of Learning to Protect Communications with Adversarial Neural Cryptography** [**https://arxiv.org/abs/1610.06918**](https://arxiv.org/abs/1610.06918)

https://github.com/rfratila/Adversarial-Neural-Cryptography




https://github.com/IBM/MAX-Adversarial-Cryptography

https://researchcode.com/code/2162819080/learning-to-protect-communications-with-adversarial-neural-cryptography/


[https://mathybit.github.io/adversarial-neural-crypto/](https://mathybit.github.io/adversarial-neural-crypto/)
[https://mc.ai/life-of-alice-bob-and-eve-with-neural-net/](https://mc.ai/life-of-alice-bob-and-eve-with-neural-net/)  ([https://github.com/VamshikShetty/adversarial-neural-cryptography-tensorflow](https://github.com/VamshikShetty/adversarial-neural-cryptography-tensorflow))

https://github.com/EXYNOS-999/Adversarial-Neural-Cryptography



LATER:

ENCRYPTION IN LATENT SPACE 

    As suggested in the future direction maybe an RL based/agent based implementation?
    Going beyond tuples and one time pad/simple ciphertext maybe equip agents with tools like DC-GANS or VAEs or other generative models?

Mathematical optimisation of the adversarial process/game:

    https://people.eecs.berkeley.edu/~sastry/pubs/Pdfs%20of%202013/RatliffCharacterization2013.pdf

Similar work: [https://arxiv.org/abs/1612.01294](https://l.messenger.com/l.php?u=https%3A%2F%2Farxiv.org%2Fabs%2F1612.01294&h=AT1l8aVUm1AgB1EM1QUnQfI4DJfFP1dxKKHn8VxrgbRMt5xkgL774xfzBkUhOs3FTZzqTecslluqzn9nhyO7mdkLklx4RcnBDkPqnyTaSoMnwV7gEtw9vE7w26mvBA)

 


**TODO**
**Implementation of Learning to Protect Communications with Adversarial Neural Cryptography** [**https://arxiv.org/abs/1610.06918**](https://arxiv.org/abs/1610.06918)

https://github.com/rfratila/Adversarial-Neural-Cryptography



https://github.com/IBM/MAX-Adversarial-Cryptography

https://researchcode.com/code/2162819080/learning-to-protect-communications-with-adversarial-neural-cryptography/


[https://mathybit.github.io/adversarial-neural-crypto/](https://mathybit.github.io/adversarial-neural-crypto/)

https://mc.ai/life-of-alice-bob-and-eve-with-neural-net/


[https://urtc.mit.edu](https://urtc.mit.edu)   Sept

https://deepmath-conference.com/


Status:
Rishabh: Implementing [https://github.com/rfratila/Adversarial-Neural-Cryptography](https://github.com/rfratila/Adversarial-Neural-Cryptography)
Compatibility with tf2.0 

https://github.com/EXYNOS-999/Adversarial-Neural-Cryptography





Avani
Pranav 

TO-DO:
[https://github.com/tensorflow/models/blob/master/research/adversarial_crypto/train_eval.py](https://github.com/tensorflow/models/blob/master/research/adversarial_crypto/train_eval.py)

https://github.com/ankeshanand/neural-cryptography-tensorflow

https://github.com/nlml/adversarial-neural-crypt


[https://mathybit.github.io/adversarial-neural-crypto](https://mathybit.github.io/adversarial-neural-crypto/)
a good resource

https://www.nuget.org/packages/CryptoN


GAN why

https://github.com/carpedm20/DCGAN-tensorflow
































MIT Undergraduate Research Technology Conference  AND/OR Top-Journa





We ask whether neural networks can learn to use secret keys to protect information from other neural networks. Specifically, we focus on ensuring confidentiality properties in a multiagent system, and we specify those properties in terms of an adversary. Thus, a system may consist of neural networks named Alice and Bob, and we aim to limit what a third neural network named Eve learns from eavesdropping on the communication between Alice and Bob. We do not prescribe specific cryptographic algorithms to these neural networks; instead, we train end-to-end, adversarially. We demonstrate that the neural networks can learn how to perform forms of encryption and decryption, and also how to apply these operations selectively in order to meet confidentiality goals.

combination of quantum computing and neural cryptography
An efficient cryptography scheme is proposed based on continuous-variable quantum neural network (CV-QNN), in which a specified CV-QNN model is introduced for designing the quantum cryptography algorithm. It indicates an approach to design a quantum neural cryptosystem which contains the processes of key generation, encryption and decryption. Security analysis demonstrates that our scheme is security. Several simulation experiments are performed on the Strawberry Fields platform for processing the classical data “Quantum Cryptography” with CV-QNN to describe the feasibility of our method. Three sets of representative experiments are presented and the second experimental results confirm that our scheme can correctly and effectively encrypt and decrypt data with the optimal learning rate 8*e* − 2 regardless of classical or quantum data, and better performance can be achieved with the method of learning rate adaption (where increase factor R1 = 2, decrease factor R2 = 0.8). Indeed, the scheme with learning rate adaption can shorten the encryption and decryption time according to the simulation results presented in Figure 12. It can be considered as a valid quantum cryptography scheme and has a potential application on quantum devices.
Thus the novel post-quantum cryptography[4](https://www.nature.com/articles/s41598-020-58928-1#ref-CR4) (including quantum cryptography([5](https://www.nature.com/articles/s41598-020-58928-1#ref-CR5),[6](https://www.nature.com/articles/s41598-020-58928-1#ref-CR6),[7](https://www.nature.com/articles/s41598-020-58928-1#ref-CR7)) which is secure against both quantum and classical computers is urgently required. Moreover, the typical scheme of quantum cryptography is implemented by combining quantum key distribution with classical “one-time pad” model[8](https://www.nature.com/articles/s41598-020-58928-1#ref-CR8),[9](https://www.nature.com/articles/s41598-020-58928-1#ref-CR9) currently, which can effectively solve the key distribution problem[10](https://www.nature.com/articles/s41598-020-58928-1#ref-CR10). While there are the problems of high key rate requirements, large key demands and consumptions in practical applications in the “one-time pad” quantum communication system. Therefore, we approach to investigate new quantum cryptography algorithms and protocols that can be implemented based on a more practical model.
Several researchers have already combined neural network with classical cryptography for the multivariate structural and nondirectional features of neural networks. In 1990, Lauria[11](https://www.nature.com/articles/s41598-020-58928-1#ref-CR11) firstly introduced the concept of cryptography based on artificial neural network (ANN). Then branches of applications and related works of cryptography with different ANN models were proposed subsequently. Network stochastic synchronization with partial information[12](https://www.nature.com/articles/s41598-020-58928-1#ref-CR12) and asymptotic, finite-time synchronization for networks with time-varying delays[13](https://www.nature.com/articles/s41598-020-58928-1#ref-CR13) provide possibilities for mutual learning between neural networks. Synchronization and learning mechanism based on neural network[14](https://www.nature.com/articles/s41598-020-58928-1#ref-CR14) prove that neural network can be trained to perform encryption and decryption operations, which is similar to the black box computing model in quantum computation[15](https://www.nature.com/articles/s41598-020-58928-1#ref-CR15).
References/Resources
[https://arxiv.org/abs/1610.06918](https://arxiv.org/abs/1610.06918)














Understanding Methodology and reproducing results.
SYSTEM ORGANIZATION 
A classic scenario in security involves three parties: Alice, Bob, and Eve. Typically, Alice and Bob wish to communicate securely, and Eve wishes to eavesdrop on their communications. Thus, the desired security property is secrecy (not integrity), and the adversary is a “passive attacker” that can intercept communications but that is otherwise quite limited: it cannot initiate sessions, inject messages, or modify messages in transit

For us, Alice, Bob, and Eve are all neural networks. We describe their structures in Sections 2.4 and 2.5. They each have parameters, which we write θA, θB, and θE, respectively. Since θA and θB need not be equal, **encryption and decryption need not be the same function** even if Alice and Bob have the same structure. 
[https://people.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf](https://people.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf)
[https://md5decrypt.net/en/Rot13/](https://md5decrypt.net/en/Rot13/)
Goal formulation and loss fxn definition are very important(!!!)
neural network architecture that was sufficient to learn mixing functions such as XOR, but that did not strongly encode the form of any particular algorithm( evolving?)
**We plan to release the source code for the experiments.**
In this paper, we demonstrate that neural networks can learn to protect communications. The learning does not require prescribing a particular set of cryptographic algorithms, nor indicating ways
of applying these algorithms: it is based only on a secrecy specification represented by the training
objectives. In this setting, we model attackers by neural networks; alternative models may perhaps
be enabled by reinforcement learning.
[https://researchcode.chttps://researchcode.com/code/2162819080/learning-to-protect-communications-with-adversarial-neural-cryptography/](https://researchcode.com/code/2162819080/learning-to-protect-communications-with-adversarial-neural-cryptography/)
[om/code/2162819080/learning-to-protect-communications-with-adversarial-neural-cryptography/](https://researchcode.com/code/2162819080/learning-to-protect-communications-with-adversarial-neural-cryptography/)
Some hacky implementations:
[https://github.com/tensorflow/models/blob/master/research/adversarial_crypto/train_eval.py](https://github.com/tensorflow/models/blob/master/research/adversarial_crypto/train_eval.py)

https://github.com/carpedm20/DCGAN-tensorflow

https://github.com/nlml/adversarial-neural-crypt


[https://mathybit.github.io/adversarial-neural-crypto/](https://mathybit.github.io/adversarial-neural-crypto/)


# Great write-up:

[https://nlml.github.io/neural-networks/adversarial-neural-cryptography/](https://nlml.github.io/neural-networks/adversarial-neural-cryptography/)

# Ideas:

As suggested in the future direction maybe an RL based/agent based implementation?
Going beyond tuples and one time pad/simple ciphertext maybe equip agents with tools like DC-GANS or VAEs or other generative models?
Have multiple adversaries.

Adversary more layers/complex than the other agents 

#  
#  
#  
#  
#  
#  
#  
#  
#  
# Learning Perfectly Secure Cryptography to Protect Communications with Adversarial Neural Cryptography

[Murilo Coutinho](https://www.ncbi.nlm.nih.gov/pubmed/?term=Coutinho%20M%5BAuthor%5D&cauthor=true&cauthor_uid=29695066),1,† [Robson de Oliveira Albuquerque](https://www.ncbi.nlm.nih.gov/pubmed/?term=de%20Oliveira%20Albuquerque%20R%5BAuthor%5D&cauthor=true&cauthor_uid=29695066),1,† [Fábio Borges](https://www.ncbi.nlm.nih.gov/pubmed/?term=Borges%20F%5BAuthor%5D&cauthor=true&cauthor_uid=29695066),2,† [Luis Javier García Villalba](https://www.ncbi.nlm.nih.gov/pubmed/?term=Garc%26%23x000ed%3Ba%20Villalba%20LJ%5BAuthor%5D&cauthor=true&cauthor_uid=29695066),3,*† and [Tai-Hoon Kim](https://www.ncbi.nlm.nih.gov/pubmed/?term=Kim%20TH%5BAuthor%5D&cauthor=true&cauthor_uid=29695066)4

https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5982701/


 
Learning to Protect Communications with Adversarial Neural Cryptography
[Martín Abadi](https://arxiv.org/search/cs?searchtype=author&query=Abadi%2C+M), [David G. Andersen](https://arxiv.org/search/cs?searchtype=author&query=Andersen%2C+D+G) (Google Brain)
We ask whether neural networks can learn to use secret keys to protect information from other neural networks. Specifically, we focus on ensuring confidentiality properties in a multiagent system, and we specify those properties in terms of an adversary. Thus, a system may consist of neural networks named Alice and Bob, and we aim to limit what a third neural network named Eve learns from eavesdropping on the communication between Alice and Bob. We do not prescribe specific cryptographic algorithms to these neural networks; instead, we train end-to-end, adversarially. We demonstrate that the neural networks can learn how to perform forms of encryption and decryption, and also how to apply these operations selectively in order to meet confidentiality goals.

----------

**Quick start for Quantum Computing**

https://www.ibm.com/quantum-computing/


[https://quantum-computing.ibm.com](https://quantum-computing.ibm.com)
[https://www.dwavesys.com/take-leap](https://www.dwavesys.com/take-leap)
[https://rigetti.com](https://rigetti.com)  I’m in touch with these guys if we can formulate a proposal then we can get a dedicated QP

https://aws.amazon.com/braket/


[https://arxiv.org/abs/1610.06918](https://arxiv.org/abs/1610.06918)
Implementation-tf: [https://github.com/ankeshanand/neural-cryptography-tensorflow](https://github.com/ankeshanand/neural-cryptography-tensorflow)
Implementation-theano:[https://github.com/nlml/adversarial-neural-crypt](https://github.com/nlml/adversarial-neural-crypt)
**Attacking RSA with NNs**
[**https://eprint.iacr.org/2016/921.pdf**](https://eprint.iacr.org/2016/921.pdf)
[**http://www.computerscijournal.org/vol2no1/approaches-in-rsa-cryptosystem-using-artificial-neural-network/**](http://www.computerscijournal.org/vol2no1/approaches-in-rsa-cryptosystem-using-artificial-neural-network/)
**Designing Quantum Cryptography Algo with QNN**

https://www.nature.com/articles/s41598-020-58928-1


**See also:** [**https://www.xanadu.ai/software/**](https://www.xanadu.ai/software/)
[https://ieeexplore.ieee.org/document/7724953](https://ieeexplore.ieee.org/document/7724953)

https://www.technologyreview.com/2019/05/30/65724/how-a-quantum-computer-could-break-2048-bit-rsa-encryption-in-8-hours/


[https://www.quantum-inspire.com](https://www.quantum-inspire.com)
 


## **Literature review and people who have worked in this sub-domain(very-few)**
## **Trackbacks for** [1610.06918](https://arxiv.org/abs/1610.06918)

[https://scholar.google.co.in/scholar?q=Adversarial+Neural+Cryptography&hl=en&as_sdt=0&as_vis=1&oi=scholart](https://scholar.google.co.in/scholar?q=Adversarial+Neural+Cryptography&hl=en&as_sdt=0&as_vis=1&oi=scholart)
[Life of Alice, Bob and Eve with Neural Net](https://arxiv.org/tb/redirect/1825974/ed029bb66) \[Towards Data Science@ towardsdatascience.com/life...\] [trackback posted Mon, 10 Dec 2018 05:05:38 UTC]
[Security and Privacy considerations in Artificial Intelligence & Machine Learning -- Part 5: When...](https://arxiv.org/tb/redirect/1825752/32f7c93ef) \[Towards Data Science@ towardsdatascience.com/secu...\] [trackback posted Mon, 5 Nov 2018 21:47:30 UTC]
[Train your Neurons on Artificial Neural Networks [featuring Keras]](https://arxiv.org/tb/redirect/1825067/d4025b90d) \[Towards Data Science@ towardsdatascience.com/trai...\] [trackback posted Fri, 29 Jun 2018 17:05:28 UTC]
[Adversarial Neural Cryptography can Solve the Biggest Friction Point in Modern AI](https://arxiv.org/tb/redirect/1825047/5054379fa) \[Towards Data Science@ towardsdatascience.com/adve...\] [trackback posted Wed, 27 Jun 2018 17:03:26 UTC]
Learning Perfectly Secure Cryptography to
Protect Communications with Adversarial
Neural Cryptography
[https://www.mdpi.com/1424-8220/18/5/1306/pdf](https://www.mdpi.com/1424-8220/18/5/1306/pdf)



----------


Neural Cryptography: From Symmetric Encryption to Adversarial Steganography Dylan Modesitt, Tim Henry, Jon Coden, and Rachel Lathe
[https://pdfs.semanticscholar.org/63e2/d6e2e467d2640765eace46885cb5c27530dc.pdf](https://pdfs.semanticscholar.org/63e2/d6e2e467d2640765eace46885cb5c27530dc.pdf)
TEACHING NEURAL NETWORKS TO HIDE DATA IN DATA Now that we have a sufficiently strong adversary, we want to see if we can design a network that will be able to hide a secret in plainsight in the presence of this adversary. We explore neural approaches for text in image, image in image, and video in video. Furthermore, we use roughly the same network for all pairings of data.


----------

 https://arxiv.org/pdf/1602.02830.pdf







----------

[https://arxiv.org/abs/1602.02672](https://arxiv.org/abs/1602.02672)

https://www.sciencedirect.com/science/article/pii/0022000084900709?via%3Dihub


[https://arxiv.org/abs/1605.07736](https://arxiv.org/abs/1605.07736)








Interesting parallel:
[https://cs.stanford.edu/people/eroberts/courses/cs181/projects/1999-00/dmca-2k/macrovision.html](https://cs.stanford.edu/people/eroberts/courses/cs181/projects/1999-00/dmca-2k/macrovision.html)
Testbed:
Twitter bots:
Alice:------> Bob 
Adversary tries to decipher, this adds constraints to the state space of the transmission of information.
Idea:
Give agent(s) lemmas and priors/basic theory about cryptography?
See also: 
Semi-supervised approach?






**Discussion points:**
Neural networks are generally not meant to be great at cryptography. Famously, the simplest neural networks cannot even compute XOR, which is basic to many cryptographic algorithms. Nevertheless, as we demonstrate, neural networks can learn to protect the confidentiality of their data from other neural networks: they discover forms of encryption and decryption, without being taught specific algorithms for these purposes.
Traditional cryptographic methods based on keys(statistical/non-differential) value addition here the function(unknown) used here is differential 
we demonstrate, neural networks can learn to protect the confidentiality of their data from other neural networks: they discover forms of encryption and decryption, without being taught specific algorithms for these purposes
**Loss Functions** Eve’s loss function is exactly as described above: the L1 distance between Eve’s guess and the input plaintext. The loss function for Alice and Bob is more complex, as indicated in Sections 2.2 and 2.3. This function has two components, related to Bob’s reconstruction error and to the eavesdropper’s success. The first component is simply the L1 distance between Bob’s output and the input plaintext. The latter component, on the other hand, is (N/2 − Eve L1 error) 2 /(N/2)2 . This definition expresses the goal, described in Section 2.3, that Eve should not do better than random guessing. Accordingly, this component is minimized when half of the message bits are wrong and half are right. We choose a quadratic formula in order to place more emphasis on making Eve have a large error, and to impose less of a penalty when Eve guesses a few bits correctly, as should happen occasionally even if Eve’s guesses are effectively random. Adopting this formulation allowed us to have a meaningful per-example loss function (instead of looking at larger batch statistics), and improved the robustness of training. Its cost is that our final, trained Alice and Bob typically allow Eve to reconstruct slightly more bits than purely random guessing would achieve. We have not obtained satisfactory results for loss functions that depend linearly (rather than quadratically) on Eve’s reconstruction error. The best formulation remains an open question.
total bits: N = 16 (plaintext + key)
target:  Eve's error > 7.3 (8 ideally in case of random guessing)
alice bob's < .5  
Out of 20 runs of NN training 6 fail ( bob's error> .5 or Eve's error < 7.3)
on success:
Retrain Eve 5 times and get Eve's error in range 4.67 to 6.97 (mean6.1)
If we somewhat arbitrarily define success as maintaining Bob’s reconstruction error at or under 0.05 bits, and requiring that Eve get at least 6 bits wrong, on average, then training succeeded half of the time (ten of twenty cases).
Although training with an adversary is often unstable (Salimans et al., 2016), we suspect that some additional engineering of the neural network and its training may be able to increase this overall success rate. With a minibatch size of only 512, for example, we achieved a success rate of only 1/3 (vs. the 1/2 that we achieved with a minibatch size of 4096). In the future, it may be worth studying the impact of minibatch sizes, and also that of other parameters such as the learning rate.

**Changes in the ciphertext induced by various plaintext/key pairs**

- key-dependent: changing the key and holding the plaintext constant results in different ciphertext output.
- plaintext-dependent, as required for successful communication.
- not simply XOR.
- the output values are often floating-point values other than 0 and 1.
- the effect of a change to either a key bit or a plaintext bit is spread across multiple elements in the ciphertext, not constrained to a single bit as it would be with XOR.
- A single-bit flip in the key typically induces significant changes in three to six of the 16 elements in the ciphertext, and smaller changes in other elements. 
- Plaintext bits are similarly diffused across the ciphertext

 **NEURAL NETWORK ARCHITECTURE AND TRAINING GOALS**
We use an augmented version of the neural network architecture of Section 2.4. The inputs first go into a new FC layer (12 inputs—eight key bits and four values—and 12 outputs); the outputs of that first layer are fed into a network with the architecture of Section 2.4. Intuitively, we chose this augmented architecture because a single FC layer should be capable of predicting D from A, B, and C, as well as making a prediction decorrelated with C; and the architecture of Section 2.4 suffices to encrypt any of the output of the first layer under the key. We therefore believed this augmented architecture would be sufficient to accomplish its task, though it may be more than is necessary to do so.
**CONCLUSION** 
In this paper, we demonstrate that neural networks can learn to protect communications. The learning does not require prescribing a particular set of cryptographic algorithms, nor indicating ways of applying these         algorithms: it is based only on a secrecy specification represented by the training objectives. In this setting, we model attackers by neural networks; alternative models may perhaps be enabled by reinforcement learning. There is more to cryptography than encryption. In this spirit, further work may consider other tasks, for example steganography, pseudorandom-number generation, or integrity checks. Finally, neural networks may be useful not only for cryptographic protections but also for attacks. While it seems improbable that neural networks would become great at cryptanalysis, they may be quite effective in making sense of metadata and in traffic analysis.
-------------------------------------------
**Comments:**
 [https://deepmind.com/research/publications/Acme](https://deepmind.com/research/publications/Acme)
Explore if there exists other frameworks for training distributed adversarial  agents.
Very basic implementation w/ limited bits and mini-batch size 
AEs and other variants(as adversary) might be interesting to see if able to reconstruct data.
ref** [https://distill.pub/2016/deconv-checkerboard/](https://distill.pub/2016/deconv-checkerboard/)

https://deepmind.com/research/publications/Acme


[https://openai.com/blog/adversarial-example-research/](https://openai.com/blog/adversarial-example-research/)
[https://arxiv.org/abs/1805.06605](https://arxiv.org/abs/1805.06605)  here
[https://openaccess.thecvf.com/content_cvpr_2018/html/Dong_Boosting_Adversarial_Attacks_CVPR_2018_paper.html](https://openaccess.thecvf.com/content_cvpr_2018/html/Dong_Boosting_Adversarial_Attacks_CVPR_2018_paper.html)
**Related work:**
Two neural networks which are trained on their mutual output bits show a novel phenomenon: The networks synchronize to a state with identical time dependent weights. It is shown how synchronization by mutual learning can be applied to cryptography: secret key exchange over a public channel.
[https://arxiv.org/pdf/cond-mat/0208453.pdf](https://arxiv.org/pdf/cond-mat/0208453.pdf)
Amazing work @CSAIL
[https://courses.csail.mit.edu/6.857/2018/project/Modesitt-Henry-Coden-Lathe-NeuralCryptography.pdf](https://courses.csail.mit.edu/6.857/2018/project/Modesitt-Henry-Coden-Lathe-NeuralCryptography.pdf)
Pros:

1. Easy to experiment/ similar to RL arch
2. Relatively less compute required.
3. Easy to change architecture.
4. Analysis can be done on every timestep
5. Relatively less knowledge of Crytography required, but interdisciplinary work .
6. Novelty of the work, interesting to discuss and get reviews from other researchers.
7. Would be cool to see the results even if it fails.
8. Classical mathematical groundwork for statistical crypto work is very well studies so can explore ideas.

Cons:

1. Lot of work into thinking the architecture and formulating the problem
2. Very few references and work/similar experiments
3. Open source code? Have to make major modifications maybe? or small modifs with significant result = publication :P

 classical cryptographic functions are generally not differentiable, so they are at odds with training by stochastic gradient descent (SGD), the main optimization technique for deep neural networks. Therefore, we would have trouble learning what to encrypt, even if we know how to encrypt.
**Current:**
Rishabh Chakrabarty    [https://notrishabh.co](https://notrishabh.co)
Avani Gupta                 [https://www.linkedin.com/in/avani17101-gupta/?originalSubdomain=in](https://www.linkedin.com/in/avani17101-gupta/?originalSubdomain=in)

**Potential:**
Rudrabha Mukhopadhyay 
[https://www.linkedin.com/in/rudrabha-mukhopadhyay-b52b86156/?originalSubdomain=in](https://www.linkedin.com/in/rudrabha-mukhopadhyay-b52b86156/?originalSubdomain=in)
Mari Sosa                 [https://www.linkedin.com/in/mari-sosa-159a87183/](https://www.linkedin.com/in/mari-sosa-159a87183/)
Chris Endemann [https://www.linkedin.com/in/chris-endemann](https://www.linkedin.com/in/chris-endemann)
Meha Kaushik        [https://www.linkedin.com/in/meha27/](https://www.linkedin.com/in/meha27/)
Anuj Rathore         [https://www.linkedin.com/in/anujrathore123/](https://www.linkedin.com/in/anujrathore123/)
Kritika Prakash         [https://www.linkedin.com/in/kritika-prakash-07057510a/](https://www.linkedin.com/in/kritika-prakash-07057510a/)
Ethan Seto         [https://www.linkedin.com/in/ethan-seto/](https://www.linkedin.com/in/ethan-seto/)
Parv Kapoor         [https://www.linkedin.com/in/parv-kapoor/](https://www.linkedin.com/in/parv-kapoor/)
Yen Low                 [https://www.linkedin.com/in/yenlow/](https://www.linkedin.com/in/yenlow/)
Mithun Nalla               ****[https://www.linkedin.com/in/mithunnallana/](https://www.linkedin.com/in/mithunnallana/) 
Kritika?

**1. Lisa Iatckova, Weill Cornell Medicine** 
**email: ani4002@med.cornell.edu** 

**2. Divya Brundavanam, Karolinska Institute** 
**email: bdivya@umich.edu** 

**3. Wei Zhang, Washington University in St. Louis** 
**email: wzzang@me.com** 
**Google Scholar, Personal Page:** [**https://scholar.google.com/citations?hl=en&user=9qru1g8AAAAJ**](https://scholar.google.com/citations?hl=en&user=9qru1g8AAAAJ)**, _** 

**4. Anish Simhal, Child Mind Institute** 
**email: aksimhal@gmail.com** 
**Google Scholar, Personal Page:** [**https://scholar.google.com/citations?hl=en&user=NefDuV0AAAAJ&view_op=list_works&sortby=pubdate**](https://scholar.google.com/citations?hl=en&user=NefDuV0AAAAJ&view_op=list_works&sortby=pubdate)**,** [**https://aksimhal.github.io/**](https://aksimhal.github.io/) ****

**5. FuTe Wong, Academia Sinica** 
**email: zuxfoucault@gmail.com** 
**Google Scholar, Personal Page: _,** [**https://github.com/zuxfoucault**](https://github.com/zuxfoucault) ****

**6. Amy Kuceyeski, Weill Cornell Medicine** 
**email: amk2012@med.cornell.edu** 
**Google Scholar, Personal Page:** [**https://scholar.google.com/citations?user=cfUvMIYAAAAJ&hl=en**](https://scholar.google.com/citations?user=cfUvMIYAAAAJ&hl=en)**,** [**https://www.cocolaboratory.com/**](https://www.cocolaboratory.com/) ****
**Potential supervisors/co-authors(Personal connections/can reach out to them, some have offered to help)**
Ida Momennejad ****[https://www.momen-nejad.org](https://www.momen-nejad.org)         
Rajan Lab           [http://labs.neuroscience.mssm.edu/project/rajan-lab/](http://labs.neuroscience.mssm.edu/project/rajan-lab/)
Balaraman Ravindran [https://www.linkedin.com/in/balaraman-ravindran-427a307/?originalSubdomain=in](https://www.linkedin.com/in/balaraman-ravindran-427a307/?originalSubdomain=in)
Adam White         [https://sites.ualberta.ca/~amw8/](https://sites.ualberta.ca/~amw8/)
Kinship Lab          [https://kinshiplab.org](https://kinshiplab.org) Winter: 
Juan Álvaro Gallego  
Zach mainen          [https://mainenlab.org](https://mainenlab.org)
Marco                  [http://zugarolab.net](http://zugarolab.net)
Wann Jiun Ma          [https://www.linkedin.com/in/wannjiun/](https://www.linkedin.com/in/wannjiun/)
Dan Goodman 
Simon B Eickhoff [http://www.neurosciences-duesseldorf.de/principal-investigators-and-junior-researchers/simon-b-eickhoff.html](http://www.neurosciences-duesseldorf.de/principal-investigators-and-junior-researchers/simon-b-eickhoff.html)

Cryptanalysis Direction 
[https://www.cs.rit.edu/~ark/students/kj4401/report.pdf](https://www.cs.rit.edu/~ark/students/kj4401/report.pdf)
A Machine Learning Approach for Cryptanalysis 
- Kowsic Jayachandiran
From the results obtained, it is evident that neural networks is a possible solution to many questions we have in the field of cryptanalysis. As such, there has been only a few methods tried in the domain of cryptanalysis since the key space of any complex cipher system is large. **The project also demonstrates that the neural network works well with one round of the Simon cipher but does not perform well when it comes to working with higher number of rounds**. This is only the first step of many more possibilities that could be explored with neural networks; with more knowledge acquired in the future about neural networks, the number of rounds could be increased and the network would be able to come up with an approximation function provided it has all the required resources for such heavy computations.
An extension of this project in the future could be a design based on fuzzy classifiers that could be used along with the neural networks to yield probabilities for 0 and 1 rather than a hard decision that a neural network does. The fuzzy classifier in itself could possibly do a brute-force search for
the key in decreasing order of key probability, resulting in less work than an exhaustive key search.  
[http://www.cs.ndsu.nodak.edu/~siludwig/Publish/papers/NaBIC2017.pdf](http://www.cs.ndsu.nodak.edu/~siludwig/Publish/papers/NaBIC2017.pdf)
This paper na?
Cryptography and Machine Learning
Ronald L. Rivest*
Laboratory for Computer Science
Massachusetts Institute of Technology
Cambridge, MA 02139
[https://people.csail.mit.edu/rivest/pubs/Riv91.pdf](https://people.csail.mit.edu/rivest/pubs/Riv91.pdf)
Other possibilities \Ve have seen some successful applications of continuous optimization techniques (such as gradient descent) to discrete learning problems; here the neural net technique of "back propagation" comes to mind. Perhaps such techniques could also be employed successfully in cryptanalytic problems.
Nothing else on the optimistic side mentioned see: effect of Information Theory on Cryptanalysis part
[https://www.csa.iisc.ac.in/~cris/resources/pdf/AP_CryptoML.pdf](https://www.csa.iisc.ac.in/~cris/resources/pdf/AP_CryptoML.pdf)
[https://crypto.iacr.org/2018/slides/goldwasser_iacr_distinguished_lecture.pdf](https://crypto.iacr.org/2018/slides/goldwasser_iacr_distinguished_lecture.pdf)
Search → Probabiliscally and Approximately Correct Learning 
The best example of black-box, end-to-end learning of the type you describe in the literature is probably Greydanus' work on *Learning the Enigma With Recurrent Neural Networks*. They achieve functional key recovery for the restricted version of Enigma they study, but require *much* more data and computing power than traditional cryptanalysis of the same mechanism would. The paper itself freely points this out; black-box, end-to-end learning to decrypt just is hard.
Links: [Paper](https://arxiv.org/abs/1708.07576) [Blog post](https://greydanus.github.io/2017/01/07/enigma-rnn/) [Code](https://github.com/greydanus/crypto-rnn) 
Comments/highlights
 *“If the output of an algorithm when interacting with the [encryption] protocol matches that of a simulator given some inputs, it ‘need not know’ anything more than those inputs”* ([Wikipedia](https://en.wikipedia.org/wiki/Black_box))
 
Outside the black box setting, one can however do a lot better. At the time of writing, the best reference I am aware of is my CRYPTO 2019 paper *Improving Attacks on Round-Reduced Speck32/64 Using Deep Learning*. The main attack of the paper breaks 11-round Speck32/64 roughly 200 times faster than the best previous cryptanalysis:
[Paper](https://ia.cr/2019/037) [Talk](https://youtu.be/weX1itU9VrM) [Code](https://www.github.com/agohr/deep_speck)
The code also contains (and the paper describes in a footnote) an easily practical attack on 12-round Speck using the same methods.
Finally, AI is not the same as machine learning and does not necessarily have to even *use* machine learning. With that in mind, e.g. *Using SMT Solvers to Automate Chosen Ciphertext Attacks* by Beck, Zinkus and Green would I think count as "using AI techniques for cryptanalysis" as well:
[Paper](https://ia.cr/2019/958) [WAC2 talk](https://youtu.be/qfCXmF11-is)
**Most notably, we demonstrate that an RNN with a 3000-unit Long Short-Term Memory (LSTM) cell can learn the decryption function of the Enigma machine**

**Only if I had a Nvidia DGX lol**
[**https://arxiv.org/abs/1708.07576**](https://arxiv.org/abs/1708.07576)

https://github.com/EXYNOS-999/crypto-rnn


Pros:

1. Interesting

Cons:

1. More compute required if the implementation is not efficient.
2. Related work hasn’t shown promising results.
3.  Need someone with deep domain knowledge in classical cryptography
4. Again traditional adversarial networks would fail as they are trained to be an “adversary” to another network, not a cryptographic function. 
5. Fuckton of literature review lol
6. Most of the cryptographical fxns used today have been proved to be NP-Hard.

**Paper Presentation Guidelines** 
[https://urtc.mit.edu/paper.pdf](https://urtc.mit.edu/paper.pdf)
The following are guidelines for presentations associated with papers at the IEEE MIT Undergraduate Research Technology Conference. Presenters are solely responsible for the creation of their presentation. 
The presentation should be based on the approved/accepted paper, but may include updates and related additional content. A paper must be presented otherwise it will not be published and archived in the IEEE Xplore database. Only listed authors may present a paper. For a successful and productive conference, all presenters should adhere to the following guidelines: 
• All presentations are to be in English. The presenter should be able to understand and respond to audience questions in English. 
• Presentations are to be 10 minutes, inclusive of 2 minutes for questions. • A Windows laptop and projector will be made available in each presentation room. Macintosh compatibility is not guaranteed. • Presenters should save their presentation in .pdf or .ppt format on a flashdrive (memory stick). Cloud-based storage (e.g. Dropbox, Google Drive) and presentation software (e.g. Prezi) should not be used as access to the internet is not guaranteed. • Arrive to your session room 15 minutes before the session begins to upload your presentation to the provided laptop. Presentation from the personal laptops may be acceptable, but compatibility with the projector cannot be guaranteed. • Presenters are reminded to dress professionally. • Presentations may be recorded and made publically available. If you do not wish to be recorded, please notify the Chair of your session prior to starting your presentation. • Contact the program committee (gimsoon@ieee.org) immediately if you are unable to attend the conference









**POSTER GUIDELINES** 
[https://urtc.mit.edu/poster.pdf](https://urtc.mit.edu/poster.pdf)
--------------------------
The following are guidelines for poster presentations associated with accepted abstract at the
IEEE MIT Undergraduate Research Technology Conference.
Poster sessions are a valuable method for undergraduate students to present research or project,
and meet with interested attendees for in-depth technical discussions. Therefore, it is important
that you display your results clearly to attract people who have an interest in your work and your paper.
You are solely responsible for the creation of your posters. The presentation should be based on
the approved/accepted abstract, but may include updates and related additional content. Only listed authors may present the poster.
For a successful and productive conference, all presenters should adhere to the following guidelines:
• Poster size 2 x 3 feet (width by height). Poster board will be provided to hold your poster, and the
poster board can be placed either on a table or an easel.
• All presentations are to be in English. You should understand and respond to audience questions in English.
• Your poster should cover the key points of your work. The ideal poster is designed to attract attention provide a brief overview of your work initiate discussion and questions
• The title of your poster should appear at the top in CAPITAL letters about 25mm (1″) high.
• The author(s) name(s) and affiliation(s) are put below the title.
• Carefully prepare your poster well in advance of the conference. There will be no time or materials available for last minute preparations at the conference. If you think you may need certain materials to repair the poster after traveling, bring them with you.
• Use color to highlight and make your poster more attractive, by using pictures, diagrams, cartoons, figures, etc., rather than only text wherever possible. 
• The smallest text on your poster should be at least 9mm (3/8″) high, and the important points should be in a larger font. Make your poster as self-explanatory as possible. This will save you time to use for discussions and questions.
• Presenters are reminded to dress professionally.
• Contact the program committee (gimsoon@ieee.org) immediately if you are unable to attend the conference. 

**OTHER DISCUSSION** 
One of questions we had five years ago was how to build better generative models. But we have been able to make huge progress in the ability to generate images after the proposals of the VAEs (variational autoencoders) and the GANs (generative adversarial networks).
We are still working on problems like disentangling underlying factors of variation, which is a question I asked ten years ago. We’ve made some progress but no that much, so it is still an open question. At that time, I was hoping the techniques we had would magically do the right thing, now I think we need some priors to push things into the right direction.
[https://medium.com/syncedreview/yoshua-bengio-on-ai-priors-and-challenges-2ebdc0a758d](https://medium.com/syncedreview/yoshua-bengio-on-ai-priors-and-challenges-2ebdc0a758d1)
Applications of Generative architectures:(Mostly applied research)
GAN Specific: [https://github.com/nashory/gans-awesome-applications](https://github.com/nashory/gans-awesome-applications)
Video Prediction
**Interpolation** 
In order to ensure reliable real-time communication, it is necessary to deal with packets that are missing when the receiver needs them. Specifically, if new audio is not provided continuously, glitches and gaps will be audible, but repeating the same audio over and over is not an ideal solution, as it produces artifacts and reduces the overall quality of the call. The process of dealing with the missing packets is called [packet loss concealment](https://en.wikipedia.org/wiki/Packet_loss_concealment) (PLC). The receiver’s PLC module is responsible for creating audio (or video) to fill in the gaps created by packet losses, excessive jitter or temporary network glitches, all three of which result in an absence of data

https://ai.googleblog.com/2020/04/improving-audio-quality-in-duo-with.html


 
[Do Deep Generative Models Know What They Don't Know](https://videos.re-work.co/videos/1790-do-deep-generative-models-know-what-they-don-t-know)
*Balaji Lakshminarayanan, Staff Research Scientist at DeepMind*
A neural network deployed in the wild may be asked to make predictions for inputs that were drawn from a different distribution than that of the training data. Generative models are widely viewed to be a solution for detecting out-of-distribution (OOD) inputs and distributional skew, as they model the density of the input features p(x). Balaji and DeepMind challenge this assumption by presenting several counter-examples. In this presentation, Balaji explains his findings including that deep generative models, such as flow-based models, VAEs and PixelCNN, which are trained on one dataset (e.g. CIFAR-10) can assign higher likelihood to OOD inputs from another dataset (e.g. SVHN). Further investigation into some of these failure modes in detail, that help us better understand this surprising phenomenon, and potentially fix them are included in the above presentation. You can see Balaji's full presentation [here](https://videos.re-work.co/videos/1790-do-deep-generative-models-know-what-they-don-t-know).

https://videos.re-work.co/videos/1790-do-deep-generative-models-know-what-they-don-t-know


[Building Generative Models Of Symptomatic Health Data for Autonomous Deep Space Missions](https://videos.re-work.co/videos/1724-building-generative-models-of-symptomatic-health-data-for-autonomous-deep-space-missions)
*Krittika D'Silva , AI Researcher at NASA Frontier Development Lab*
In this presentation, Krittika talks on her work at NASA FDL in which she examined how AI can be used to support medical care in space. Future NASA deep space missions will require advanced medical capabilities, including continuous monitoring of astronaut vital signs to ensure optimal crew health. Also discussed in this presentation is biosensor data collected from NASA analog missions can be used to train AI models to simulate various medical conditions that might affect astronauts. Other topics covered include the future of AI and space medicine, continuous monitoring using wearables, symptomatic data v unsymptomatic data and more. See the full presentation from Krittika [here](https://videos.re-work.co/videos/1724-building-generative-models-of-symptomatic-health-data-for-autonomous-deep-space-missions).

https://videos.re-work.co/videos/1724-building-generative-models-of-symptomatic-health-data-for-autonomous-deep-space-missions


 
Evolutionary Generative Adversarial Networks
Show affiliations

- [Wang, Chaoyue](https://ui.adsabs.harvard.edu/#search/q=author:%22Wang%2C+Chaoyue%22&sort=date%20desc,%20bibcode%20desc);
-  [Xu, Chang](https://ui.adsabs.harvard.edu/#search/q=author:%22Xu%2C+Chang%22&sort=date%20desc,%20bibcode%20desc);
-  [Yao, Xin](https://ui.adsabs.harvard.edu/#search/q=author:%22Yao%2C+Xin%22&sort=date%20desc,%20bibcode%20desc);
-  [Tao, Dacheng](https://ui.adsabs.harvard.edu/#search/q=author:%22Tao%2C+Dacheng%22&sort=date%20desc,%20bibcode%20desc)

Abstract
Generative adversarial networks (GAN) have been effective for learning generative models for real-world data. However, existing GANs (GAN and its variants) tend to suffer from training problems such as instability and mode collapse. In this paper, we propose a novel GAN framework called evolutionary generative adversarial networks (E-GAN) for stable GAN training and improved generative performance. Unlike existing GANs, which employ a pre-defined adversarial objective function alternately training a generator and a discriminator, we utilize different adversarial training objectives as mutation operations and evolve a population of generators to adapt to the environment (i.e., the discriminator). We also utilize an evaluation mechanism to measure the quality and diversity of generated samples, such that only well-performing generator(s) are preserved and used for further training. In this way, E-GAN overcomes the limitations of an individual adversarial training objective and always preserves the best offspring, contributing to progress in and the success of GANs. Experiments on several datasets demonstrate that E-GAN achieves convincing generative performance and reduces the training problems inherent in existing GANs.



- [https://ui.adsabs.harvard.edu/abs/2018arXiv180300657W/abstract](https://ui.adsabs.harvard.edu/abs/2018arXiv180300657W/abstract)
- [https://cds.cern.ch/record/2256878?ln=en](https://cds.cern.ch/record/2256878?ln=en)
- [https://cds.cern.ch/search?f=490__a&p=IML%20Machine%20Learning%20Workshop](https://cds.cern.ch/search?f=490__a&p=IML%20Machine%20Learning%20Workshop)
- [https://mickypaganini.github.io/atlasML.html](https://mickypaganini.github.io/atlasML.html)

AUTOENCODERS AND VARIANTS 

