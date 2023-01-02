import utils as ut
import numpy as np
# import sys

def transcribe_emails(emails):
    vocab_list = ut.getVocabDict()
    transcribed_emails = np.zeros((len(emails), len(vocab_list)))

    for email in range(len(emails)):
        # ignore non utf-8 characters
        email_contents = open(emails[email], 'r', encoding='utf-8', errors='ignore').read()
        curr_email = ut.email2TokenList(email_contents)
        # sys.exit("Error message")

        for word in range(len(curr_email)):
            if curr_email[word] in vocab_list:
                word_id = vocab_list[curr_email[word]]  - 1
                transcribed_emails[email, word_id] = 1

    return transcribed_emails

def check_spam_svm():
    easy_ham_mails = transcribe_emails(ut.obtainAllDirectoryFiles('./data_spam/easy_ham/'))
    hard_ham_mails = transcribe_emails(ut.obtainAllDirectoryFiles('./data_spam/hard_ham/'))
    spam_mails =  transcribe_emails(ut.obtainAllDirectoryFiles('./data_spam/spam/'))

    print('E_H len:', len(easy_ham_mails))
    print('H_H len:', len(hard_ham_mails))
    print('S len:', len(spam_mails))

def main(training_sys='svm'):
    
    if training_sys == 'svm':
        check_spam_svm()
    elif training_sys == 'nn':
        return 1
    elif training_sys == 'log':
        return 0

if __name__ == '__main__':
    main()