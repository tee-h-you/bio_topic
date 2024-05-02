import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}

search_result_page = ["https://www.nature.com/search?journal=ncomms,%20commsbio,%20nmeth&article_type=research&subject=computational-biology-and-bioinformatics&date_range=2024-2024&order=date_desc",
                      "https://www.nature.com/search?journal=ncomms,%20commsbio,%20nmeth&article_type=research&subject=computational-biology-and-bioinformatics&date_range=2023-2023&order=date_desc",
                      "https://bmcbioinformatics.biomedcentral.com/articles?tab=keyword&searchType=journalSearch&sort=PubDate&volume=25",
                      "https://bmcbioinformatics.biomedcentral.com/articles?tab=keyword&searchType=journalSearch&sort=PubDate&volume=24"
                      ]
def abstract_extractor(journal_name, link):
    try:
        response = requests.get(link, headers=headers)
        if response.status_code == 200:
            content = response.text
            soup = BeautifulSoup(content, 'html.parser')
            if journal_name == "nature":
                abstract_tag = soup.find('div', {'id': 'Abs1-content'})
                title_tag = soup.find('h1', class_ = 'c-article-title')
            if journal_name == "bmcbioinformatics":
                abstract_tag = soup.find ('div', {'id': 'Abs1-content'} )
                title_tag = soup.find('h1', class_ = 'c-article-title')

            if abstract_tag:
                abstract = abstract_tag.get_text(separator='\n')
                title = title_tag.get_text(separator='\n')
                return title.strip(), abstract.strip()
            else:
                return "", "Abstract is not found."
        else:
            return "","Page is not found"
    except Exception as e:
        return "",str(e)

def search_result_extractor(journal_name,link):

    try:
        links = []
        response = requests.get(link, headers=headers)
        if response.status_code == 200:
            content = response.text
            soup = BeautifulSoup(content, 'html.parser')
            if journal_name == 'nature':
                for link_ in soup.find_all ('a', href=re.compile ('^/articles/')):
                    links.append (link_['href'])
                return links
            if journal_name == 'bmcbioinformatics':
                for link_ in soup.find_all ("a", attrs={"data-test": "title-link"}):
                    links.append (link_['href'])
                return links
        else:
            return "Page is not found."
    except Exception as e:
        print(e)

def write_BMCbio_data_to_csv( num_page, search_result_first_page, csv_name):

    all_links = []
    for i in range(1,num_page+1):
        page = f'{search_result_first_page}&page={i}'
        links = search_result_extractor('bmcbioinformatics',page)
        for link in links:
            all_links.append(f'https://bmcbioinformatics.biomedcentral.com{link}')
    abstracts = []
    titles = []
    for link in all_links:
        title, abstract = abstract_extractor('bmcbioinformatics',link)
        print('---------------------------')
        print(title)
        print(abstract)
        if (abstract != "Abstract is not found.") and (abstract != "Page is not found."):
            abstracts.append(abstract)
            titles.append(title)
    print(f'Found {len(all_links)} papers and {len(abstracts)} abstracts\n')

    df = pd.DataFrame({'Title': titles, 'Abstract': abstracts} )
    df.to_csv(csv_name)
def write_Nature_data_to_csv( num_page, search_result_first_page, csv_name):
    all_links = []
    for i in range (1, num_page + 1):
        page = f'{search_result_first_page}&page={i}'
        links = search_result_extractor ('nature', page)
        for link in links:
            all_links.append (f'https://nature.com{link}')
    abstracts = []
    titles = []
    for link in all_links:
        title, abstract = abstract_extractor ('nature', link)
        print ('---------------------------')
        print (title)
        print (abstract)
        if (abstract != "Abstract is not found.") and (abstract != "Page is not found."):
            abstracts.append (abstract)
            titles.append (title)
    print (f'Found {len (all_links)} papers and {len (abstracts)} abstracts\n')

    df = pd.DataFrame ({'Title': titles, 'Abstract': abstracts})
    df.to_csv (csv_name)
def main():
    #Nature journal
    write_Nature_data_to_csv(7,search_result_page[0], "nature_abstracts_2024.csv")
    write_Nature_data_to_csv(16,search_result_page[1], "nature_abstracts_2023.csv")

    #BMC Bioinformatics
    write_BMCbio_data_to_csv(4, search_result_page[2], "BMC-bioinformatics_abstracts_2024.csv")
    write_BMCbio_data_to_csv(10, search_result_page[3], "BMC-bioinformatics_abstracts_2023.csv")

main()
