#!/usr/bin/python
# -*- coding: utf-8 -*-
# routines for processing biomedical NER and RE
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_KERAS'] = '1'

from ie_utils.ie_conversion import *
from ie_utils.ie_options import *

from re_utils.re_docsets import *
from re_utils.re_docset import *
from re_utils.re_docsnt import *
from re_utils.re_instance import *
#
from el_utils.el_docsets import *
from el_utils.el_docset import *
from el_utils.el_docsnt import *

# from ner_utils.ner_span_docsets import *
# from ner_utils.ner_span_docset import *
# from ner_utils.ner_span_docsnt import *
# from ner_utils.ner_span_instance import *

def get_task_doc_classes(tcfg, task):
    # class initialization
    if task == 'ner':
        # if 'Span' in tcfg.model_name:
        #     doc_classes = nerSpanDocSets, nerSpanDocSet, nerSpanDocSnt, nerSpanInst
        # else:
        #     doc_classes = nerDocSets, nerDocSet, nerDocSnt, nerInst
        doc_classes = nerDocSets, nerDocSet, nerDocSnt, nerInst
    elif task == 're':
        if 'Mrc' in tcfg.model_name:
            doc_classes = reMrcDocSets, reMrcDocSet, reMrcDocSnt, reInst
        else:
            doc_classes = reDocSets, reDocSet, reDocSnt, reInst
    elif task == 'ure':
        doc_classes = reDocSets, ureDocSet, ureDocSnt, reInst
    elif task == 'nre':  # n-ary RE
        doc_classes = nreDocSets, nreDocSet, nreDocSnt, nreInst
    else:   # 'el', entity linking, instance is entity mention
        doc_classes = elDocSets, elDocSet, elDocSnt, EntityMention
    return doc_classes


def review_ner_corpus(op, tcfg, task, wdir, cpsfiles, cpsfmts):
    # get the classes
    DocSets, DocSet, DocSnt, Inst = get_task_doc_classes(tcfg, task)
    # prepare the docsets
    nersets = DocSets(task, wdir, cpsfiles, cpsfmts)
    nersets.prepare_corpus_filesets(op, tcfg, DocSet, DocSnt, Inst)
    # statistics on entity mentions
    ccounts = nersets.calculate_docsets_instance_statistics(tcfg)
    nersets.output_docsets_instance_statistics(nersets.dcfg.ent_typedict, ccounts, 'Entity Mentions',
                                               logfile='EntityMentions.cnt')
    if not tcfg.bertID:
        sfiles = ['{}_{}_voc.txt'.format(task, cf) for cf in cpsfiles]
        combine_word_voc_files(wdir, sfiles, '{}_voc.txt'.format(task), verbose=True)
    # save the instances of the last file
    nersets.filesets[-1].print_docset_instances(filename=wdir+'/test_ent.txt', level='docu', verbose=tcfg.verbose)
    return


def review_el_corpus(op, tcfg, task, wdir, cpsfiles, cpsfmts):
    # get the classes
    DocSets, DocSet, DocSnt, Inst = get_task_doc_classes(tcfg, task)
    # prepare the docsets
    elsets = DocSets(task, wdir, cpsfiles, cpsfmts)
    elsets.prepare_corpus_filesets(op, tcfg, DocSet, DocSnt, Inst)
    # filter entity candidates by name searching, save candidates to *.ent.cdd
    elsets.filter_docsets_entity_names(op, tcfg)
    return


def review_re_corpus(op, tcfg, task, wdir, cpsfiles, cpsfmts, nary):
    DocSets, DocSet, DocSnt, Inst = get_task_doc_classes(tcfg, task)
    resets = DocSets(task, wdir, cpsfiles, cpsfmts, nary=nary)
    resets.prepare_corpus_filesets(op, tcfg, DocSet, DocSnt, Inst)
    if tcfg.mrcID: return
    #
    task = resets.ftask
    dcfg = resets.dcfg
    # statistics on entity mentions
    if cpsfmts[0] != 'i':  # only for sentence-level and document-level
        levels = ('sent', 'docu')
        ccounts = resets.calculate_docsets_entity_mention_statistics(tcfg, levels)
        resets.output_docsets_instance_statistics(dcfg.ent_typedict, ccounts, 'Entity Mentions', levels,
                                                  logfile='reEntityMentions.cnt')
    # statistics on relation mentions
    ccounts = resets.calculate_docsets_instance_statistics(tcfg)
    logfile = '{}RelationMentions.cnt'.format('u' if task == 'ure' else ('n' if task == 'nre' else ''))
    resets.output_docsets_instance_statistics(dcfg.rel_typedict, ccounts, 'Relation Mentions', logfile=logfile)
    if not tcfg.bertID:
        sfiles = ['{}_{}_voc.txt'.format(task, cf) for cf in cpsfiles]
        combine_word_voc_files(wdir, sfiles, '{}_voc.txt'.format(task), verbose=True)
    # statistics for MRC-based RE
    # if task == 're' and tcfg.mrcID:
    #     DocSets, DocSet, DocSnt, Inst = nerMrcDocSets, nerMrcDocSet, nerMrcDocSnt, nerMrcInst
    #     nersets = resets.convert_docsets_stask_dtask(op, tcfg, DocSets, DocSet, DocSnt, Inst, task='ner', stask='re')
    #     ccounts = nersets.calculate_docsets_instance_statistics(tcfg)
    #     nersets.output_docsets_instance_statistics(tcfg.ent_typedict, ccounts, 'Relation-Entity Mentions', logfile=logfile)
    # save the instances of the last file
    tfilename = '{}/{}_{}.txt'.format(wdir, resets.filesets[-1].id, task)
    resets.filesets[-1].print_docset_instances(filename=tfilename, level='inst', verbose=tcfg.verbose)
    return


def browse_ie_corpus(op, tcfg, task, wdir, cpsfiles, cpsfmts, nary=2, bfileno=-1):
    """
    :param bfileno: the index of the corpus file to be browsed
    :return: None
    """
    # class initialization
    DocSets, DocSet, DocSnt, Inst = get_task_doc_classes(tcfg, task)
    # prepare docsets
    iesets = DocSets(task, wdir, [cpsfiles[bfileno]], [cpsfmts[bfileno]], nary=nary)
    iesets.prepare_corpus_filesets(op, tcfg, DocSet, DocSnt, Inst)
    # browse units
    if task == 'el':    # browse entity bank
        tcfg.entbank.browse_units()
    else:   # browse docset
        ie = DocsetExplorer(tcfg, cpsfiles[bfileno], iesets.filesets[0])
        ie.browse_units(tcfg)
    return


def train_eval_corpus(op, tcfg, task, wdir, cpsfiles, cpsfmts, nary=2, seed=0, folds=None):
    """
    train, evaluate and predict on a corpus for NER, RE, ...
    :return: None
    """
    set_my_random_seed(seed)

    DocSets, DocSet, DocSnt, Inst = get_task_doc_classes(tcfg, task)
    iesets = DocSets(task, wdir, cpsfiles, cpsfmts, nary=nary)
    iesets.prepare_corpus_filesets(op, tcfg, DocSet, DocSnt, Inst)
    #
    # if task == 're' and tcfg.mrcID:  # convert RE to MRC-based NER
    #     iesets.train_eval_docsets_as_mrc_ner(op, tcfg, folds)
    # else:
    #     iesets.train_eval_docsets(op, tcfg, folds)
    iesets.train_eval_docsets(op, tcfg, folds)
    return


def train_eval_ner_corpus(op, tcfg, task, wdir, cpsfiles, cpsfmts, nary=2, seed=0, folds=None):
    """
    train, evaluate and predict on a corpus for NER
    :return: None
    """
    set_my_random_seed(seed)

    # DocSets, DocSet, DocSnt, Inst = nerMrcDocSets, nerMrcDocSet, nerMrcDocSnt, nerMrcInst
    DocSets, DocSet, DocSnt, Inst = nerDocSets, nerDocSet, nerDocSnt, nerInst
    # prepare DISEASE docsets
    dwdir, dcpsfiles, dcpsfmts = 'NCBI', ('train', 'dev', 'test'), 'aaa'
    ncbi_sets = DocSets(task, dwdir, dcpsfiles, dcpsfmts, nary=nary)
    ncbi_sets.prepare_corpus_filesets(op, tcfg, DocSet, DocSnt, Inst)
    ncbi_sets.print_docsets_instance_numbers()
    # prepare CHEMICAL corpus
    cwdir, ccpsfiles, ccpsfmts = 'CHEMD', ('dev', 'test'), 'aaa'
    chemd_sets = DocSets(task, cwdir, ccpsfiles, ccpsfmts, nary=nary)
    chemd_sets.prepare_corpus_filesets(op, tcfg, DocSet, DocSnt, Inst)
    chemd_sets.print_docsets_instance_numbers()
    # prepare target docsets
    nersets = DocSets(task, wdir, cpsfiles, cpsfmts, nary=nary)
    for i, cpsfile in enumerate(cpsfiles):
        fileset = DocSet(task, wdir=wdir, id=cpsfile, fmt=cpsfmts[i])
        nersets.filesets.append(fileset)
    #
    nersets.input_docsets_instances(ncbi_sets)
    nersets.input_docsets_instances(chemd_sets)
    nersets.print_docsets_instance_numbers()
    # prepare for training
    nersets.load_cfg_dict_file(tcfg)  # task cfg file --> tcfg.*
    for i, fileset in enumerate(nersets.filesets):
        fileset.prepare_docset_dicts_features(op, tcfg)
    #
    nersets.train_eval_docsets(op, tcfg, folds)
    return


task2suff = {'ner': 'ent', 're': 'rel', 'ure': 'urel', 'nre': 'nrel', 'el': 'lnk'}


def main(op, task, wdir, cpsfiles, cpsfmts, tcfg, nary=2, seed=0, folds=range(1)):
    tcfg.word_vector_path = './glove/glove.6B.100d.txt'
    tcfg.bert_path = './bert-model/biobert-pubmed-v1.1'
    # tcfg.task = task.upper()

    if wdir in ('SMV', 'CONLL2003'): tcfg.bert_path = './bert-model/bert-base-uncased'
    # two entities with different types are required for some relations
    if wdir in ('CPR', 'CDR'):
        tcfg.diff_ent_type = 1
        tcfg.head_ent_types = ['CHEM']  # for MRC-based RE
    # SemEval task 8
    if wdir == 'SMV':
        tcfg.avgmode = 'macro'
        tcfg.mark_ent_pair = ''  # already has entity marks
    # assign tokenizer, default is tokenize_bio_sentence
    if task == 'ner':
        tcfg.sent_tokenizer = tokenize_ner_bio_sentence
        tcfg.case_sensitive = 1
    # sentence simplification for RE/URE
    if task in ('re', 'ure', 'nre'): tcfg.sent_simplify = 1
    if task == 'el':
        tcfg.word_vector_path = None
        tcfg.case_sensitive = 1
        tcfg.ent_bank_name = 'CTD_diseases2019.csv'
    #
    tcfg.save_epoch_model = 1 if tcfg.bertID else 0
    # predicted file suffix
    tcfg.task_file_suff = task2suff[task]
    # nrel1, nrel2, nrel
    if task == 'nre':
        tcfg.rel_arg_num = nary
        tcfg.task_file_suff = '{}{}'.format(task2suff[task], nary)
    #
    load_bert_word_dict(tcfg)
    if 'f' in op:     # format the corpus
        convert_bio_corpus(tcfg, wdir, cpsfiles)
    elif 'r' in op:  # prepare word vocabulary
        # tcfg.model_name, tcfg.bertID = 'Lstm', False    # default model name for review
        if task == 'ner':  # NER
            review_ner_corpus(op, tcfg, task, wdir, cpsfiles, cpsfmts)
        elif task in ('re', 'ure', 'nre'):
            review_re_corpus(op, tcfg, task, wdir, cpsfiles, cpsfmts, nary)
        elif task == 'el':
            review_el_corpus(op, tcfg, task, wdir, cpsfiles, cpsfmts)
    elif 'b' == op:
        # tcfg.model_name, tcfg.bertID = 'Lstm', False  # default model name for browse
        browse_ie_corpus(op, tcfg, task, wdir, cpsfiles, cpsfmts, bfileno=-1)
    elif wdir == 'BioNER':  # unified NER model
        train_eval_ner_corpus(op, tcfg, task, wdir, cpsfiles, cpsfmts, nary, seed, folds)
    else:
        train_eval_corpus(op, tcfg, task, wdir, cpsfiles, cpsfmts, nary, seed, folds)
    # save logging files
    file_list2line(tcfg.log_lines, tcfg.log_file_name, verbose=0)

if __name__ == '__main__':
    elist = ('GENE', 'CHEM', 'DISE', 'BPRO')    # entity types to be replaced/blinded with PLACEHOLDERS
    options = OptionConfig(model_name='Bert', batch_size=10, epochs=15, bepo=-1,
                           save_epoch_model=1, sele_best_metric='F1', max_seq_len=128,
                           valid_ratio=0, neg_samp_ratio=0, verbose=3, trn_mdl_fit=1,  # use fit to train model
                           add_loss_model=0, trunc_train_batch=0, ensemble_classifier=0,
                           fold_rand=1, fold_valid=1, fold_num=10, fold_num_run=1,
                           bld_ent_types=elist, bld_ent_num=0, bld_ent_mode=2, # 0-None, 1-type, 2-typename
                           mark_ent_pair='#@#@',  # ''-None, '#@#@', '##@@'
                           diff_ent_type=0, mask_nest_ent=0,  # 0-不, 1-内层,2-外层,-1-没有嵌套实体
                           case_sensitive=0, dist_embedding=0, cnn_kernel_size=[3, 4, 5],
                           pred_test_ent=0, browse_level='docu', elabel_schema='BIEO',
                           calc_sent_metric=1, calc_docu_metric=1, out_rst_level='inst',
                           bert_model_impl='ZJG', max_span_len=6,
                           # trained_model_file='NCBI/ner_MrcBert_e5_f0.hdf5',
                           el_SF_LF=0, el_combo_thres=0, el_filter_num=30)
    # options.add_option('--pos_embedding', dest='pos_embedding', default=1)
    tcfg, _ = options.parse_args()
    # NER
    # main('v', 'ner', 'CONLL2003', ('train', 'dev', 'test'), 'iii', tcfg, folds=range(1))
    # main('tv', 'ner', 'JNLPBA', ('train', 'dev'), 'ii', tcfg)
    # main('tv', 'ner', 'BC2GM', ('train', 'test'), 'ss', tcfg, seed=0, folds=range(2,3))
    # main('v', 'ner', 'CHEMD', ('train', 'dev', 'test'), 'aaa', tcfg, folds=range(1))
    # main('v', 'ner', 'NCBI', ('train', 'dev', 'test'), 'aaa', tcfg, seed=0, folds=range(1))
    # main('r', 'ner', 'NCBI', ('test',), 'aaa', tcfg, seed=0, folds=range(1))
    # main('tv', 'el', 'NCBI', ('train', 'dev', 'test'), 'aaa', tcfg, seed=0, folds=range(1))
    # main('tv', 'ner', 'CDR', ('train', 'dev', 'test'), 'aaa', tcfg, seed=0, folds=range(1))
    # main('v', 'ner', 'CEMP', ('train', 'dev'), 'aa', 'Bert')
    # main('v', 'ner', 'GPRO',('train', 'dev'), 'aa', tcfg)
    # main('r', 'ner', 'LINN', ('train',), 'f', tcfg)
    # main('tv', 'ner', 'S800', ('train',), 'a', tcfg)
    # main('vb', 'ner', 'CPR', ('train', 'dev', 'test'), 'aaa', tcfg, folds=range(1))
    #
    # main('tv', 'ner', 'BioNER', ('train', 'test'), 'ii', tcfg, seed=0, folds=range(1))
    # RE
    # main('tv', 'ure', 'GENIA', ('train',), 'i', tcfg, folds=range(1))  # SMV, mark_ent_pair=''
    # main('r', 're', 'SMV', ('train', 'test'), 'ii', tcfg, folds=range(1))  # SMV, mark_ent_pair=''
    # main('r', 're', 'PPI', ('train',), 'i', tcfg, seed=0, folds=range(1))       # PPI
    main('tvp', 're', 'CPR', ('train', 'dev', 'test'), 'aaa', tcfg, seed=0, folds=range(5))     # CPR, diff_ent_type=1
    # main('r', 're', 'CPR', ('test',), 'aaa', tcfg, seed=0, folds=range(1))  # CPR, diff_ent_type=1
    # main('tv', 're','GAD', ('train',), 'i',  tcfg)
    # main('tv', 're', 'EUADR', ('train',), 'i', tcfg)
    # for i in range(4, 10):
    # main('v', 'nre', 'BEL', ('train', 'test'), 'ss', tcfg, nary=3, seed=0, folds=range(1))  # BEL
    # main('tv', 're', 'BEL', ('train', 'test'), 'ss', tcfg, seed=0, folds=range(1))  # BEL-binary relation
    # main('tvp', 'ure', 'BEL_GENIA', ('train', 'test',), 'ss', tcfg, seed=0, folds=range(5))  # BEL-binary relation
    # main('v', 'ure', 'BEL', ('train', 'test'), 'ss', tcfg, seed=0, folds=range(1))  # BEL-unary relation
    exit(0)
    """ corpus format:
    i - instance-level like CoNLL2003, JNLPBA2004
    s - sentence-level like BC2GM, text: sntid, sentence 
    a - abstract-level like CPR, text: pmid, title and abstract
    f - full text like LINNAEUS
    """
    """ func list:
    f - format, convert original corpus files to standard files like *.txt, *.ent, *.rel
    r - review the corpus, prepare json config file, combine word vocabularies
    t - train using the whole corpora , including train, dev and test sets.
    v - evaluate on the last file, performance is reported. 
    p - predict on the last file, no performance is reported
    tv - train using the corpus except the last file, evaluate on the last one.
         if only one file exists, cross-validation will be performed. (FOLD_NUM, FOLD_NUM_RUN) 
    tp - train using the corpus except the last file, predict on the last one
    b - browse different level data units, i.e. docus, sents and insts, independently or after op 'v' 
    """
    """ model_filename for non Cross-Validation, PS: for CV, these 3 parameters are ignored.
        '{}/{}_{}_e{}_f{}.hdf5'.format(wdir, task, model_name, epoch, fold), like SMV/re_Bert_e3_f0.hdf5, or
        '{}/{}_{}_{}_e{}_f{}.hdf5'.format(wdir, stask, task, model_name, epoch, fold), like SMV/ve_trg_ner_Bert_e3_f0.hdf5
    bepo: best epoch, valid for 'tvp' to indicate which epoch model to apply, 0 for best, -1 for last
    seed: valid only for 't' to set the initial seed.
    folds: valid for multi-runs 'tvp' with the same seed, or to indicate folds of models for ensemble classification
    """
