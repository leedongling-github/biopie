import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_KERAS'] = '1'

from bel_utils.bel_docset import belDocSet

from sbel_utils.sbel_docsets import *
from sbel_utils.sbel_docset import *
from sbel_utils.sbel_docsnt import *

from ie_utils.ie_options import *


def generate_sbel_statements(tcfg, wdir, cpsfile):
    # 从语料文件的BEL语句和实体文件中产生SBEL语句，保存到文件 *.sbel.[gold|re.gold], 其中re模式不考虑功能
    # 将产生的SBEL语句再合并成相应的BEL语句，并保存到文件 *.bel.[gold|re.gold]
    belset = belDocSet(wdir, id=cpsfile, strict=tcfg.eval_strict, verbose=tcfg.verbose)
    belset.generate_docset_sbel_statements(tcfg)
    return


def evaluate_bel_statements(tcfg, wdir, cpsfile, folds=None):
    """
    评估语料文件中的BEL语句在各个层面上的预测情况, *.bel vs *.bel.[gold|re.gold|prd|re.prd]
    :param wdir, cpsfile:
    """
    # gold 结尾用于评估BEL->SBEL的损失率
    if tcfg.eval_bel_suff == 'gold':    # 评估重组后的BEL语句
        evaluate_gold_bel_statements(wdir, cpsfile, tcfg)
        return

    # sbelfile = '{}/{}.sbel.{}'.format(wdir, cpsfile, tcfg.eval_bel_suff)
    # combine RE and URE files to SBEL file for tcfg.decompose
    # if tcfg.sbel_decompose:  # predicted
    #     brfile = '{}/{}_sbel.rel.prd'.format(wdir, cpsfile)
    #     urfile = '{}/{}_sbel.urel.prd'.format(wdir, cpsfile)
    #     combine_sbel_predicted_results(brfile, urfile, sbelfile, tcfg.verbose)
    # convert SBEL to BEL
    # entfile = '{}/{}.ent'.format(wdir, cpsfile)
    # tbelfile = '{}/{}.bel.{}'.format(wdir, cpsfile, tcfg.eval_bel_suff)  # [train|test].bel.[gold|re|prd]
    # convert_sbel_to_bel(sbelfile, entfile, tbelfile, inte_func=True, combine_comp=True, verbose=tcfg.verbose)

    # load entity mentions
    belset = belDocSet(wdir=wdir, id=cpsfile, strict=tcfg.eval_strict, verbose=tcfg.verbose)
    belset.load_entity_mentions(tcfg.verbose)
    # merge SBEL to BEL statements w/o functions
    if not folds:  folds = [None]
    for fold in folds:
        fsuf = '.f{}'.format(fold) if fold is not None else ''
        sbelfile = '{}/{}.sbel.{}{}'.format(wdir, cpsfile, tcfg.eval_bel_suff, fsuf)
        belset.load_sbel_statements(sbelfile, is_gold=False, verbose=tcfg.verbose)  # load into test_sbels
        # merge to BEL statements w/o functions from belset.test_sbels,
        # save to *.bel.[eval_bel_suff], *.bel.rel.[eval_bel_suff]
        belset.merge_docset_sbel_to_bel_statements(tcfg, is_gold=False, fold=fold)
        # evaluate *.bel.[eval_bel_suff] vs *.bel, output results to *.log and *.rst
        belset.evaluate_docset_bel_statements(tcfg)
        # clear BEL statements
        for _, snt in belset.sntdict.items():
            snt.gold_bels, snt.test_bels, snt.test_rels = [], [], []
        #
    return


def evaluate_gold_bel_statements(wdir, cpsfile, tcfg):
    # load entity mentions
    belset = belDocSet(wdir=wdir, id=cpsfile, strict=tcfg.eval_strict, verbose=tcfg.verbose)
    belset.load_entity_mentions(tcfg.verbose)
    # evaluate *.bel.gold against *.bel
    belset.evaluate_docset_bel_statements(tcfg)
    return


def browse_bel_statements(tcfg, wdir, cpsfile, fold):
    """
    浏览语料文件：显示和匹配某一语料中句子的标准和预测的BEL和SBEL语句
    :param tcfg:
    :param wdir:
    :param cpsfile:
    :param fold:
    """
    belset = belDocSet(wdir, id=cpsfile, strict=tcfg.eval_strict, verbose=tcfg.verbose)
    # copy *.bel.prd.f0 --> *.bel.prd, *.sbel.prd.f0 --> *.sbel.prd
    if tcfg.eval_bel_suff != 'gold':
        for bsuff in ('bel', 'sbel'):
            sbelfile = '{}/{}.{}.{}.f{}'.format(wdir, cpsfile, bsuff, tcfg.eval_bel_suff, fold)
            tbelfile = '{}/{}.{}.{}'.format(wdir, cpsfile, bsuff, tcfg.eval_bel_suff)
            safe_copy_file(sbelfile, tbelfile, tcfg.verbose)
    # compare *.bel vs *.bel.[eval_bel_suff], *.sbel vs *.sbel.[eval_bel_suff]
    belset.browse_docset_bel_statements(tcfg)
    return


def review_sbel_corpus(op, tcfg, task, wdir, cpsfiles, cpsfmts):
    sbelsets = sBelDocSets(task, wdir, cpsfiles, cpsfmts)
    sbelsets.prepare_corpus_filesets(op, tcfg, sBelDocSet, sBelDocSnt, sBelInst)
    # statistics on SBEL statements, relations ans functions
    ccounts = sbelsets.calculate_docsets_instance_statistics(tcfg)
    rcounts = [[count[0] for count in counts] for counts in ccounts]  # relation
    fcounts = [[count[1] for count in counts] for counts in ccounts]  # function
    sbelsets.output_docsets_instance_statistics(tcfg.rel_typedict, rcounts, 'Relations', logfile='sBelStatements.cnt')
    sbelsets.output_docsets_instance_statistics(tcfg.func_typedict, fcounts, 'Functions', logfile='sBelStatements.cnt')
    # statistics on entity mentions
    levels = ('sent', 'docu')
    # sbelsets.filesets[0].create_entity_type_dict(tcfg, levels[-1])
    ccounts = sbelsets.calculate_docsets_entity_mention_statistics(tcfg, levels)
    sbelsets.output_docsets_instance_statistics(tcfg.ent_typedict, ccounts, 'Entity Mentions', levels,
                                                logfile='sBelStatements.cnt')
    #
    if tcfg.sbel_decompose:
        # binary relation statistics
        bresets = sbelsets.convert_docsets_stask_dtask(op, tcfg, reDocSets, reDocSet, reDocSnt, reInst, 're', 'sbel')
        ccounts = bresets.calculate_docsets_instance_statistics(tcfg)
        bresets.output_docsets_instance_statistics(tcfg.rel_typedict, ccounts, 'Relation Mentions',
                                                   logfile='brelations.cnt')
        # unary relation statistics
        uresets = sbelsets.convert_docsets_stask_dtask(op, tcfg, reDocSets, ureDocSet, ureDocSnt, reInst, 'ure', 'sbel')
        ccounts = uresets.calculate_docsets_instance_statistics(tcfg)
        uresets.output_docsets_instance_statistics(tcfg.rel_typedict, ccounts, 'uRelation Mentions',
                                                   logfile='urelations.cnt')
    return


# train, evaluate and predict NER on a corpus
# op: t-train with training files, v-validate [-1], p-predict [-1]
def train_eval_corpus(op, tcfg, task, wdir, cpsfiles, cpsfmts, bepos=None, seed=0, folds=None):
    set_my_random_seed(seed)

    sbelsets = sBelDocSets(task, wdir, cpsfiles, cpsfmts)
    sbelsets.prepare_corpus_filesets(op, tcfg, sBelDocSet, sBelDocSnt, sBelInst)
    #分离模型
    tcfg.bepo = bepos[0]
    if tcfg.sbel_decompose:
        bresets = uresets = None
        # binary relation extraction
        if tcfg.sbel_binary:
            tcfg.task_file_suff = 'rel'
            bresets = sbelsets.convert_docsets_stask_dtask(op, tcfg, reDocSets, reDocSet, reDocSnt, reInst, 're', 'sbel')
            bresets.train_eval_docsets(op, tcfg, folds)
        # unary(一元) relation extraction
        if tcfg.sbel_unary:
            tcfg.class_weight = np.array([1, 1, 1, 1, 1, 1, 2])
            tcfg.task_file_suff = 'urel'
            tcfg.bepo = bepos[1]
            uresets = sbelsets.convert_docsets_stask_dtask(op, tcfg, reDocSets, ureDocSet, ureDocSnt, reInst, 'ure', 'sbel')
            uresets.train_eval_docsets(op, tcfg, folds)
        # reload SBEL cfg file, which may be overridden by sub-tasks
        sbelsets.load_cfg_dict_file(tcfg)
        # merge binary/unary to SBEL statements, save to file like *.[sbel|bel].[prd|gold.prd]
        if tcfg.sbel_merge or (tcfg.sbel_binary and tcfg.sbel_unary):
            # 将二元关系和功能(可选, 如果tcfg.sbel_unary==1)合并为SBEL语句，计算SBEL语句性能，输出SBEL预测结果。
            for fold in folds:
                # is_gold: use the gold binary and unary relations
                sbelsets.filesets[-1].merge_docset_rel_to_sbel_statements(op, tcfg, bresets, uresets, is_gold=False, fold=fold)
                # sbelsets.filesets[-1].merge_docset_sbel_statements(op, tcfg, bresets, uresets, is_gold=False, fold=fold)
    # 联合训练模型
    else:
        tcfg.task_file_suff = 'sbel'
        sbelsets.train_eval_docsets(op, tcfg, folds)
    return


def main(op, task, wdir, cpsfiles, cpsfmts, tcfg=None, bepos=None, seed=0, folds=range(1)):
    tcfg.word_vector_path = './glove/glove.6B.100d.txt'
    tcfg.bert_path = './bert-model/biobert-pubmed-v1.1'
    #加载bert词表，用于后续的分词片
    load_bert_word_dict(tcfg)
    if 'g' in op:  # generate SBEL from BEL statements, back to BEL again
        generate_sbel_statements(tcfg, wdir, cpsfiles[-1])  # only gold and re
    elif 'w' == op:  # browse evaluation results
        browse_bel_statements(tcfg, wdir, cpsfiles[-1], folds[-1])
    elif 'e' in op:  # evaluate *.bel against *.bel.[gold|prd]
        evaluate_bel_statements(tcfg, wdir, cpsfiles[-1], folds)
    elif 'r' in op:  # prepare word vocabulary
        # tcfg.model_name, tcfg.bertID = 'Lstm', False
        review_sbel_corpus(op, tcfg, task, wdir, cpsfiles, cpsfmts)
    elif any([ch in op for ch in 'tvp']):
        train_eval_corpus(op, tcfg, task, wdir, cpsfiles, cpsfmts, bepos, seed, folds)


if __name__ == '__main__':
    elist = ('GENE', 'CHEM', 'DISE', 'BPRO')  # entity types to be replaced/blinded with PLACEHOLDER
    # batch_size = 2 before'20210522'
    options = OptionConfig(model_name='Bert', batch_size=6, epochs=3,
                           save_epoch_model=1, sele_best_metric='F1', max_seq_len=100,
                           valid_ratio=0.1, verbose=3,
                           fold_num=10, fold_num_run=1,
                           add_loss_model=0, cnn_kernel_size=[3, 4, 5],
                           pred_test_ent=0, #1表示采用stage1方式，0表示采用stage2方式
                           calc_docu_metric=0, pred_file_fold=1, #pred_file_fold 用于在每个预测结果后面加上*.f0,*.f1.. 避免不同轮次的预测文件被覆盖
                           bld_ent_types=elist, diff_ent_type=0,
                           mark_ent_pair='#@#@', bld_ent_mode=2, #bld_ent_mode 盲化方式对应的名称 0-None, 1-entity type like 'GENE', 2-entity typename like 'chemical'
                           case_sensitive=0, sent_simplify=1)

    options.add_option('--sbel_decompose', dest='sbel_decompose', default=1, help='decompose SBEL task')
    options.add_option('--sbel_binary', dest='sbel_binary', default=0, help='binary RE for SBEL')
    options.add_option('--sbel_unary', dest='sbel_unary', default=1, help='unary RE for SBEL')
    options.add_option('--sbel_merge', dest='sbel_merge', default=0, help='merge SBEL after prediction')
    # eval_bel_suff[gold, prd]: gold-recast gold BEL, prd-predicted BEL, [gold, prd] gold:用于评估SBEL到BEL的转换损失
    options.add_option('--eval_bel_suff', dest='eval_bel_suff', default='prd', help='suffix of evaluated BEL file')
    options.add_option('--eval_strict', dest='eval_strict', default=False, help='strict evaluation')
    options.add_option('--sbel_uniq_func', dest='sbel_uniq_func', default=1, help='unique function for SBEL generation')
    # options for joint learning
    options.add_option('--share_func_weights', dest='share_func_weights', default=1, help='share FD weights')
    #
    (tcfg, _) = options.parse_args()
    # SBEL
    # main('b', 'bel', 'BEL_error', ('train', 'test'), 'ss', tcfg, bepos=[1, 1], seed=0, folds=range(5))  # SBEL
    # main('vp', 'sbel', 'BEL', ('train', 'test'), 'ss', tcfg, bepos=[-1, -1], seed=0, folds=range(5))  # SBEL
    # main('vp', 'sbel', 'BEL', ('train', 'test'), 'ss', tcfg, bepos=[2, 2], seed=0, folds=range(5))  # SBEL
    main('tvp', 'sbel', 'BEL', ('train', 'test'), 'ss', tcfg, bepos=[2, 2], seed=0, folds=range(5))  # SBEL
    # main('e', 'sbel', 'BEL', ('train', 'test'), 'ss', tcfg, bepos=[-1, -1], seed=0, folds=range(5))  # SBEL
    # main('vp', 'sbel', 'BEL2', ('train', 'test'), 'ss', tcfg, bepos=[-1, -1], folds=range(1))  # SBEL
    # main('tv', 'sbel', 'BEL', ['train'], 's', tcfg, bepos=[-1], folds=range(1))  # SBEL
    exit(0)
    """ op:
    f-format, r-review, t-train, v-validate, p-predict
    e: evaluate gold BEL statements against predicted, *.bel vs. *.bel.[eval_bel_suff]
    w: browse gold and predicted BEL, SBEL statements, *.bel vs. *.bel.[eval_bel_suff], *.sbel vs. *.sbel.[eval_bel_suff] 
    g: generate SBEL relations from and back to BEL statements, save to *.sbel.gold, *.sbel.re.gold, *.bel.gold, *.bel.re.gold 
    tvp: when 'decompose' is 1, (binary=1, unary=1, eval_bel_suff='prd', sbel_merge=1)
    PS: 在导入外部的rel或urel预测文件(test.rel.prd.f0/test.urel.prd.f0)时，应将文件名拷贝为test_sbel.rel.prd.f0等，
        并设置：sbel_decompose=1, sbel_binary=0, sbel_unary=0, sbel_merge=1, 
        然后运行v功能合并成SBEL语句，再用e功能评估BEL性能。   
    """
