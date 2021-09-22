import pdfx
import spacy
import nltk

# TRANSFORMING ALL PDFS INTO TEXT

cookies_policy = pdfx.PDFx('data/Facebook/TargetCompanySourced/CookiesPolicy.pdf')
cookies_policy = cookies_policy.get_text()
#print(cookies_policy)

data_policy = pdfx.PDFx('data/Facebook/TargetCompanySourced/DataPolicy.pdf')
data_policy = data_policy.get_text()
#print(data_policy)

gen_info_privacy = pdfx.PDFx('data/Facebook/TargetCompanySourced/General Info ProtectingPrivacyAndSecurity.pdf')
gen_info_privacy = gen_info_privacy.get_text()
#print(gen_info_privacy)

open_source_privacy_policy = pdfx.PDFx('data/Facebook/TargetCompanySourced/OpenSourcePrivacyPolicy.pdf')
open_source_privacy_policy = open_source_privacy_policy.get_text()
#print(open_source_privacy_policy)

terms_of_service = pdfx.PDFx('data/Facebook/TargetCompanySourced/TermsOfService.pdf')
terms_of_service = terms_of_service.get_text()
#print(terms_of_service)

# COMPUTING STATS

nlp = spacy.load('en_core_web_sm')

doc_cookies = nlp(cookies_policy)
tokens = [t for t in doc_cookies.text.split()]

freq = nltk.FreqDist(tokens)
print('Size of Lexicon:', len(freq))
#freq.values.sort()


for key, val in freq.items():
    print(str(key) + ':' + str(val))
freq.plot(20, cumulative=False)


keywords = open('data/Keywords/Privacy KeyWords Formatted.docx.txt', "r", encoding='utf-8-sig')
keywords = keywords.read()
keywords = keywords.split(", ")
print(keywords)
