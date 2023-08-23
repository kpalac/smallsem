

ABOUT


SmallSem is a simple module and CLI application for extracting features/keywords from and summarizing text. 

The keywords are supposed to be characteristic of a document and used for findong similar documents etc. It was aimed to be simple, 
reasonably fast and accurate enough to be usable in other projects.

It makes use of Xapian database to index vocabulary from a language and then use frequencues to classify them as interesting. 
Word pairs are also used if the cooccurr in a document.

Language models' xapian indexes must be unzipped to the same folder to be functional (archives named **_index.zip).

New languages can be added by modifying generator_en.py script and by training a new Xapian DB on a corpus from a gicen language.


You can learn new texts by using SmallSemTrainer class or command:
   
    smallsem.py --lang=[SOME LANGUAGE SYMBOL] --learn-from-dir [DIRECTORY WITH PLAINTEXT]

Text provided should be in plaintext. The bigger the database the more accurate extraction is.
Language data is stored in separate folder.


You can extract keywords from a text file by command:

    smallsem.py --keywords [TEXT_FILE]

You can summarize text using:

    smallsem.py --level=[1..100] --summarize [TEXT_FILE]

If vocabulary DB is not present for a language, simple dictionaries will be used.


SmallSem also has a simple language detection feature to choose from present languages using a text sample.

Feel free to modify and play around :)


CONTACT: Karol Pałac, palac.karol@gmail.com






