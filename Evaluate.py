import pandas as pd
import re

import matplotlib.pyplot as plt
import seaborn as sns

from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary


BERTopic_topics = [
    ['protein', 'structure', 'proteins', 'structures', 'binding', 'structural', 'complexes', 'design', 'prediction',
     'molecular'],
    ['sequencing', 'assembly', 'genome', 'variant', 'data', 'reads', 'calling', 'results', 'tools', 'sequence'],
    ['singlecell', 'cell', 'data', 'scrnaseq', 'integration', 'batch', 'datasets', 'cells', 'methods', 'analysis'],
    ['drug', 'prediction', 'drugs', 'graph', 'learning', 'feature', 'dtis', 'model', 'molecular', 'methods'],
    ['mass', 'spectra', 'spectrometry', 'data', 'search', 'analysis', 'proteomics', 'metabolomics', 'peptide', 'msms'],
    ['chromatin', 'regulatory', 'enhancers', 'transcription', 'gene', 'dna', 'genes', 'methylation', 'expression',
     'cells'],
    ['model', 'data', 'models', 'feature', 'patients', 'diabetes', 'learning', 'performance', 'proposed', 'prediction'],
    ['gwas', 'variants', 'genetic', 'association', 'traits', 'ancestry', 'studies', 'genomewide', 'causal', 'method'],
    ['genomes', 'metagenomic', 'binning', 'microbial', 'plasmids', 'bacterial', 'host', 'sequencing', 'species',
     'resistance'],
    ['spatial', 'transcriptomics', 'tissue', 'spatially', 'technologies', 'st', 'data', 'tissues', 'datasets', 'cell'],
    ['imaging', 'images', 'segmentation', 'microscopy', 'image', 'optical', 'cell', 'deep', 'pipeline', 'learning'],
    ['prognostic', 'immune', 'patients', 'hcc', 'prognosis', 'survival', 'lncrnas', 'risk', 'immunotherapy', 'cox'],
    ['transmission', 'malaria', 'tb', 'epidemic', 'countries', 'cases', 'vaccine', 'covid19', 'vaccination',
     'sarscov2'],
    ['reaction', 'kinetic', 'metabolic', 'reactions', 'enzymatic', 'models', 'enzyme', 'networks', 'design',
     'turnover'],
    ['cancer', 'mutations', 'copy', 'sequencing', 'somatic', 'fusions', 'alterations', 'number', 'data', 'mutational'],
    ['cryoem', 'maps', 'cryoelectron', 'macromolecules', 'resolution', 'density', 'structural', 'microscopy',
     'complexes', 'structure'],
    ['microbiome', 'microbial', 'gut', 'species', 'growth', 'niches', 'taxa', 'metabolic', 'analysis', 'colonization'],
    ['brain', 'neurons', 'networks', 'cortical', 'network', 'activity', 'behavior', 'representations', 'neural',
     'neuronal'],
    ['biomedical', 'language', 'information', 'models', 'text', 'papers', 'extraction', 'nlp', 'literature',
     'concepts'],
    ['sarscov2', 'viral', 'variants', 'virus', 'covid19', 'mutations', 'viruses', 'clades', 'chikungunya',
     'vaccination'],
    ['images', 'segmentation', 'image', 'medical', 'augmentation', 'ultrasound', 'training', 'performance', 'learning',
     'deep'],
    ['cell', 'cancer', 'adenocarcinoma', 'cells', 'tumor', 'sc', 'heterogeneity', 'mucinous', 'microenvironment',
     'tme'],
    ['mirnadisease', 'associations', 'mirnas', 'diseases', 'network', 'lncrnas', 'prediction', 'model', 'graph',
     'potential']]

top2vec_topics = [
    ['enhancers', 'differentiation', 'transcription', 'accessibility', 'chromatin', 'developmental', 'regulatory',
     'transcriptional', 'stem', 'elements', 'transcriptomic'],
    ['files', 'calling', 'format', 'sv', 'software', 'bioinformatics', 'users', 'tools', 'easy', 'user', 'interactive'],
    ['sars', 'cov', 'virus', 'transmission', 'vaccine', 'infection', 'viruses', 'antibody', 'antibodies', 'viral',
     'host'],
    ['drug', 'drugs', 'attention', 'graph', 'graphs', 'representations', 'learn', 'convolutional', 'representation',
     'target', 'task'],
    ['batch', 'scrna', 'integration', 'clustering', 'single', 'seq', 'cell', 'omics', 'correction', 'condition',
     'latent'],
    ['diagnosis', 'diabetes', 'medical', 'ml', 'machine', 'recall', 'proposed', 'accuracy', 'decision', 'forest',
     'classification'],
    ['cryo', 'electron', 'em', 'conformational', 'membrane', 'structures', 'conformation', 'residues', 'complexes',
     'simulations', 'microscopy'],
    ['gwas', 'traits', 'genetic', 'loci', 'trait', 'causal', 'association', 'ancestry', 'variants', 'statistics',
     'wide'],
    ['imaging', 'microscopy', 'image', 'segmentation', 'images', 'resolution', 'labeled', 'tracking', 'noise',
     'supervised', 'modalities'],
    ['biomarkers', 'methylation', 'biomarker', 'prognostic', 'diagnosis', 'patient', 'cohort', 'hcc', 'patients',
     'clinical', 'cancer'],
    ['spatial', 'spatially', 'transcriptomics', 'st', 'technologies', 'transcriptomic', 'tissue', 'tissues', 'resolved',
     'cell', 'resolution'],
    ['enzymes', 'reactions', 'enzyme', 'reaction', 'metabolic', 'metabolism', 'numbers', 'growth', 'engineering',
     'family', 'studied'],
    ['folding', 'alphafold', 'complexes', 'protein', 'structures', 'native', 'structure', 'proteins', 'residues',
     'predictions', 'energy'],
    ['assembly', 'assemblies', 'reads', 'read', 'errors', 'reference', 'long', 'sequencing', 'quality', 'genomes',
     'novo'],
    ['prognostic', 'cox', 'prognosis', 'immunotherapy', 'risk', 'hcc', 'lncrnas', 'immune', 'survival', 'signature',
     'tcga'],
    ['ms', 'spectra', 'spectrometry', 'mass', 'proteomics', 'search', 'metabolomics', 'peptides', 'peptide', 'library',
     'ion'],
    ['mirna', 'mirnas', 'circrna', 'lncrna', 'rnas', 'mrna', 'associations', 'lncrnas', 'diseases', 'disease', 'been']]

LDA_topics = [['kidney', 'glycan', 'scinterpreter', 'scdreamer', 'ggcoverage', 'fos', 'ssrna', 'insnet', 'icp', 'hla'],
              ['data', 'cell', 'protein', 'based', 'methods', 'analysis', 'model', 'gene', 'cancer', 'single'],
              ['lipid', 'pockets', 'explanations', 'texttt', 'bacteriocin', 'predictors', 'ibd', 'trs', 'nl', 'tss'],
              ['asd', 'pdac', 'tme', 'drosophila', 'wound', 'stromal', 'transcriptomes', 'polyadenylation', 'mm',
               'carbonylation'],
              ['lipid', 'viral', 'ddp', 'oa', 'life', 'syngr2', 'hpc', 'il', 'colitis', 'binning'],
              ['alphafold', 'gpt', 'discrete', 'msa', 'multimer', 'gbc', 'trigon', 'gfi1b', 'amps', 'biolegato'],
              ['cryo', 'lncrnas', 'prognostic', 'hcc', 'em', 'sv', 'ancestry', 'lncrna', 'risk', 'patients'],
              ['differentiation', 'mirna', 'mirnas', 'enhancer', 'enhancers', 'ddi', 'st', 'elements', 'cell',
               'epigenetic'],
              ['gems', 'organoids', 'symptoms', 'rb', 'women', 'ecdna', 'quaternary', 'crc', 'microglia', 'her2']]


def topic_diversity(topic_lists):
    all_words = []
    if len(topic_lists) ==0:
        return 0
    else:
        for topic in topic_lists:
            for word in topic:
                if word not in all_words:
                    all_words.append(word)
        diversity_score = len(all_words) / (len(topic_lists)*len(topic_lists[0]))
        return diversity_score
def topic_coherence(topics, abstracts):
    tokenizer = lambda s: re.findall ('\w+', s.lower ())
    abstracts_ = [tokenizer (a) for a in abstracts]
    word2id = Dictionary (abstracts_)

    cm = CoherenceModel (topics=topics,
                         texts=abstracts_,
                         coherence='c_v',
                         dictionary=word2id)

    coherence_per_topic = cm.get_coherence_per_topic ()

    #Average Coherence
    sum = 0
    for line in coherence_per_topic:
        sum += line
    mean = sum / len (coherence_per_topic)
    print(f'Average coherence: {mean}\n')

    data_topic_score = pd.DataFrame (data=zip (topics, coherence_per_topic), columns=['Topic', 'Coherence'])
    data_topic_score = data_topic_score.set_index ('Topic')
    fig, ax = plt.subplots (figsize=(3, 8))
    ax.set_title ("Topics coherence\n $C_v$")
    sns.heatmap (data=data_topic_score, annot=True, square=True,
                 cmap='Reds', fmt='.2f',
                 linecolor='black', ax=ax)
    plt.yticks (rotation=0)
    ax.set_xlabel ('')
    ax.set_ylabel ('')
    plt.show ()
def main():
    BMCbio_2024 = pd.read_csv ("BMC-Bioinformatics_abstracts_2024.csv")
    BMCbio_2023 = pd.read_csv ("BMC-Bioinformatics_abstracts_2023.csv")
    nature_2024 = pd.read_csv ("nature_abstracts_2024.csv")
    nature_2023 = pd.read_csv ("nature_abstracts_2023.csv")
    all_abs = [nature_2024, BMCbio_2024, nature_2023, BMCbio_2023]
    abstracts = pd.concat (all_abs)
    abstracts = abstracts['Abstract']

    div_bert = topic_diversity (BERTopic_topics)
    div_top2vec = topic_diversity (top2vec_topics)
    div_lda = topic_diversity (LDA_topics)
    print (f'{div_bert} {div_top2vec} {div_lda}\n')

    topic_coherence(LDA_topics, abstracts)
    topic_coherence(top2vec_topics, abstracts)
    topic_coherence(BERTopic_topics, abstracts)

