from dotenv import load_dotenv
import sys
import os
from huggingface_hub import InferenceClient

classification=False
similarity=False
summarization=True
featureExtraction=False
fillmask=False

load_dotenv("C:\\Learning\\AI\\Key\\Api-key.txt")
sys.path.append('../')
HF_TOKEN=os.getenv("HUGGINGFACEHUB_API_TOKEN")

client = InferenceClient(
    provider="hf-inference",
    api_key=HF_TOKEN,
)

def textClassification (client, text="I like you. I love you"):
    classificationList = client.text_classification(text,
#       model="ProsusAI/finbert",  simple model with three levels of classification
        model="tabularisai/multilingual-sentiment-analysis", #advanced model with five levels of classification
    )
    return classificationList


def textSimilarity(client, source_sentence = "How am I doing?", other_sentences = ["I'm doing great"]):
    similarityList = client.sentence_similarity(
        source_sentence,
         other_sentences=other_sentences,
         model="google/embeddinggemma-300m")
    return similarityList


def textSummarization(client, inputSentence = "This is a dummy sentence."):
    summarizationResult = client.summarization(
        inputSentence,
        model="google/pegasus-large"
    )
    return summarizationResult


def featureExtraction(client, inputSentence = "This is a dummy sentence."):
    featureExtractionResult = client.feature_extraction(inputSentence,
        model="BAAI/bge-small-en-v1.5"
    )
    return featureExtractionResult


def fillMask(client, inputSentence = "The capital of France is <mask>."):
    fillMaskResult = client.fill_mask(
        inputSentence,
        model="FacebookAI/xlm-roberta-base"
    )
    return fillMaskResult

if (classification):
    print("Doing Text Classification Example:")
    classificationList = textClassification(client, text="Kodai is a beautiful place to visit in summer.")
    for classificationOutput in classificationList:
        print(classificationOutput['label'] + f" Score => {classificationOutput['score']}")

if (similarity):
    print("Doing Text Similarity Example:")
    source_sentence = "The quick brown fox jumps over the lazy dog."
    other_sentences = [
        "The quick brown fox jumps over the lazy dog",
        "A fast dark-colored fox leaps over a sleepy canine.", 
        "My name is Anand.",
        "வேகமான பழுப்பு நரி சோம்பேறி நாயின் மீது குதிக்கிறது.",
        "Thattanukku sattai pottal kuttai paiyan kattaiyal adipan. Avan yar."
    ]
    similarityList = client.sentence_similarity(source_sentence, other_sentences, model="google/embeddinggemma-300m")
    print("Source Sentence: \"" + source_sentence + "\"")
    for pos in range(len(similarityList)):
        print("Sentence: \"" + other_sentences[pos] + "\" + Score -> " f"{similarityList[pos]}")

if (summarization):
    print("Doing Text Summarization Example:")
    inputSentence = """Tamil is a Dravidian language natively spoken by the Tamil people of South Asia. It is one of the longest-surviving classical languages in the world, attested since 300BCE
Tamil was the lingua franca for early maritime traders in South India, with Tamil inscriptions found outside of the Indian subcontinent, such as Indonesia, Thailand, and Egypt. The language has a well-documented history with literary works like Sangam literature, consisting of over 2,000 poems. Tamil script evolved from Tamil Brahmi, and later, the vatteluttu script was used until the current script was standardized. The language has a distinct grammatical structure, with agglutinative morphology that allows for complex word formations.
Tamil is the official language of the state of Tamil Nadu and union territory of Puducherry in India. It is also one of the official languages of Sri Lanka and Singapore. Tamil-speaking diaspora communities exist in several countries across the world. Tamil was the first to be recognized as a classical language of India by the Central Government in 2004.
The earliest extant Tamil literary works and their commentaries celebrate the Pandiyan Kings for the organization of long-term Tamil Sangams, which researched, developed and made amendments to the Tamil language. Although the name of the language which was developed by these Tamil Sangams is mentioned as Tamil, the period when the name 'Tamil' came to be applied to the language is unclear, as is the precise etymology of the name. The earliest attested use of the name is found in Tholkappiyam, which is dated as early as the 2nd century BCE. The Hathigumpha inscription, inscribed around a similar period (150 BCE) by Kharavela, the Jain king of Kalinga, also refers to a Tamira Samghatta (Tamil confederacy).
The Samavayanga Sutra, dated to the 3rd century BCE contains a reference to a Tamil script named 'Damili'.
Southworth suggests that the name comes from tam-miḻ > tam-iḻ "self-speak", or 'our own speech'. Kamil Zvelebil suggests an etymology of tam-iḻ, with tam meaning 'self' or "one's self", and "-iḻ" having the connotation of "unfolding sound". Alternatively, he suggests a derivation of tamiḻ < tam-iḻ < *tav-iḻ < *tak-iḻ, meaning in origin "the proper process (of speaking)".However, this is deemed unlikely by Southworth due to the contemporary use of the compound 'centamiḻ', which means refined speech in the earliest literature.
The Tamil Lexicon of the University of Madras defines the word "Tamil" as "sweetness". SV Subramanian suggests the meaning "sweet sound", from tam – "sweet" and il – "sound".
David Shulman cites Cuntaramurti's Tevaram, in which he writes to Shiva, "Do you know proper Tamil?" and ascribes it the meaning "Do you know how to behave properly as a male lover should? Can you understand the hints and implicit meaning that a proficient lover ought to be able to decipher?" He also states that at some point in history, Tamil meant something like "knowing how to love", in a poetic sense, and that to "know Tamil" could also mean "to be a civilized being".
Tamil belongs to the southern branch of the Dravidian languages, a family of around 26 languages native to the Indian subcontinent. It is also classified as being part of a Tamil language family that, alongside Tamil proper, includes the languages of about 35 ethno-linguistic groups such as the Irula and Yerukula languages.
The closest major relative of Tamil is Malayalam; the two began diverging around the 9th century CE. Although many of the differences between Tamil and Malayalam demonstrate a pre-historic divergence of the western dialect, the process of separation into a distinct language, Malayalam, was not completed until sometime in the 13th or 14th century.
Additionally Kannada is also relatively close to the Tamil language and shares the format of the formal ancient Tamil language. While there are some variations from the Tamil language, Kannada still preserves a lot from its roots. As part of the southern family of Indian languages and situated relatively close to the northern parts of India, Kannada also shares some Sanskrit words, similar to Malayalam. Many of the formerly used words in Tamil have been preserved with little change in Kannada. This shows a relative parallel to Tamil, even as Tamil has undergone some changes in modern ways of speaking."""

    summarizationResult = textSummarization(client, inputSentence)
    print(summarizationResult.summary_text)



if(featureExtraction):
    print("Doing Feature Extraction Example:")
    inputSentence = "Kudichiduda Kaipulla. Innuma mulichiruka thoonnnggggggg"
    featureExtractionList = featureExtraction(client, inputSentence)
    for pos in range(len(featureExtractionList)):
        print(f"Feature Vector {pos}: {featureExtractionList[pos]}")


if (fillmask):
    print("Doing Fill-Mask Example:")
    inputSentence="I went for fishing at the <mask>"
    print("Input Sentence: \"" + inputSentence + "\"")
    fillMaskResultList = fillMask (client, inputSentence)
#    print(fillMaskResult)
    for fillMaskResult in fillMaskResultList:
        print(f"Option: \"{fillMaskResult['sequence']}\" with score {fillMaskResult['score']}")
#    for pos in range(len(fillMaskResult)):
#        print(f"Option {pos+1}: {fillMaskResult[pos]['sequence']} with score {fillMaskResult[pos]['score']}")