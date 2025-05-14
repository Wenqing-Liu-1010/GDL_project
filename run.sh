# ### Peptides-Struct
# python main.py   --cfg configs/peptides-struct/peptides-struct-s2gnn.yaml   out_dir tests/results/peptides-struct-s2gnn   wandb.use False   seed 11  # we use 11
# python main.py   --cfg configs/peptides-struct/peptides-struct-s2gnn.yaml   out_dir tests/results/peptides-struct-s2gnn   wandb.use False   seed 12  # we use 12
# python main.py   --cfg configs/peptides-struct/peptides-struct-s2gnn.yaml   out_dir tests/results/peptides-struct-s2gnn   wandb.use False   seed 13  # we use 13
# python main.py   --cfg configs/peptides-struct/peptides-struct-s2gnn.yaml   out_dir tests/results/peptides-struct-s2gnn   wandb.use False   seed 14  # we use 14
# python main.py   --cfg configs/peptides-struct/peptides-struct-s2gnn.yaml   out_dir tests/results/peptides-struct-s2gnn   wandb.use False   seed 15  # we use 15
# python main.py   --cfg configs/peptides-struct/peptides-struct-s2gnn.yaml   out_dir tests/results/peptides-struct-s2gnn   wandb.use False   seed 16  # we use 16
# python main.py   --cfg configs/peptides-struct/peptides-struct-s2gnn.yaml   out_dir tests/results/peptides-struct-s2gnn   wandb.use False   seed 17  # we use 17
# python main.py   --cfg configs/peptides-struct/peptides-struct-s2gnn.yaml   out_dir tests/results/peptides-struct-s2gnn   wandb.use False   seed 18  # we use 18
# python main.py   --cfg configs/peptides-struct/peptides-struct-s2gnn.yaml   out_dir tests/results/peptides-struct-s2gnn   wandb.use False   seed 19  # we use 19
# python main.py   --cfg configs/peptides-struct/peptides-struct-s2gnn.yaml   out_dir tests/results/peptides-struct-s2gnn   wandb.use False   seed 20  # we use 20

# python main.py   --cfg configs/peptides-struct/peptides-struct-gcnconv.yaml   out_dir tests/results/peptides-struct-gcnconv   wandb.use False   seed 11  # we use 11
# python main.py   --cfg configs/peptides-struct/peptides-struct-gcnconv.yaml   out_dir tests/results/peptides-struct-gcnconv   wandb.use False   seed 12  # we use 12
# python main.py   --cfg configs/peptides-struct/peptides-struct-gcnconv.yaml   out_dir tests/results/peptides-struct-gcnconv   wandb.use False   seed 13  # we use 13
# python main.py   --cfg configs/peptides-struct/peptides-struct-gcnconv.yaml   out_dir tests/results/peptides-struct-gcnconv   wandb.use False   seed 14  # we use 14
# python main.py   --cfg configs/peptides-struct/peptides-struct-gcnconv.yaml   out_dir tests/results/peptides-struct-gcnconv   wandb.use False   seed 15  # we use 15
# python main.py   --cfg configs/peptides-struct/peptides-struct-gcnconv.yaml   out_dir tests/results/peptides-struct-gcnconv   wandb.use False   seed 16  # we use 16
# python main.py   --cfg configs/peptides-struct/peptides-struct-gcnconv.yaml   out_dir tests/results/peptides-struct-gcnconv   wandb.use False   seed 17  # we use 17
# python main.py   --cfg configs/peptides-struct/peptides-struct-gcnconv.yaml   out_dir tests/results/peptides-struct-gcnconv   wandb.use False   seed 18  # we use 18
# python main.py   --cfg configs/peptides-struct/peptides-struct-gcnconv.yaml   out_dir tests/results/peptides-struct-gcnconv   wandb.use False   seed 19  # we use 19
# python main.py   --cfg configs/peptides-struct/peptides-struct-gcnconv.yaml   out_dir tests/results/peptides-struct-gcnconv   wandb.use False   seed 20  # we use 20


# ### Arxiv-Year
# #python main.py   --cfg configs/arxiv-year/arxiv-year-dirgcn-s2gnn.yaml   out_dir tests/results/arxiv-year-default   wandb.use False   seed 1  # we use 1..10
# ### Peptides-Func,这个有问题，其他的能跑
# # python main.py   --cfg configs/peptides-func/peptides-func-s2gnn.yaml   out_dir tests/results/peptides-func-default   wandb.use False   seed 1  # we use 1..10
# # ###cluster
# # python main.py   --cfg configs/cluster/cluster-gcn-s2gnn-100k.yaml   out_dir tests/results/cluster-default   wandb.use False   seed 1  # we use 1..10

# # ###custom-cluster
# # python main.py   --cfg configs/custom-cluster/custom-cluster-gmm-s2gnn.yaml   out_dir tests/results/custom-cluster-default   wandb.use False   seed 1  # we use 1..10

# # ###over-squashing
# # python main.py   --cfg configs/over-squashing/over-squashing-spec.yaml   out_dir tests/results/over-squashing-default   wandb.use False   seed 1  # we use 1..10


# # ###associative-recall
# # python main.py   --cfg configs/associative-recall/associative-recall-s2gnn.yaml   out_dir tests/results/associative-recall-default   wandb.use False   seed 1  # we use 1..10

# # ##distance regression
# # python main.py   --cfg configs/source-dist/source-dist-dag-dirgcn-s2gnn-undir.yaml   out_dir tests/results/source-dist-default   wandb.use False   seed 1  # we use 1..10


#!/bin/bash

LOG="peptides_time_log.txt"
# echo "" > $LOG  # 清空旧日志

# for seed in {11..20}
# do
#   echo "Running s2gnn with seed $seed" | tee -a $LOG
#   START=$(date +%s)
#   python main.py --cfg configs/peptides-struct/peptides-struct-s2gnn.yaml \
#     out_dir tests/results/peptides-struct-s2gnn wandb.use False seed $seed
#   END=$(date +%s)
#   echo "Time: $((END - START)) seconds" | tee -a $LOG
#   echo "---------------------------------------" >> $LOG
# done

# for seed in {11..20}
# do
#   echo "Running gcnconv with seed $seed" | tee -a $LOG
#   START=$(date +%s)
#   python main.py --cfg configs/peptides-struct/peptides-struct-gcnconv.yaml \
#     out_dir tests/results/peptides-struct-gcnconv wandb.use False seed $seed
#   END=$(date +%s)
#   echo "Time: $((END - START)) seconds" | tee -a $LOG
#   echo "---------------------------------------" >> $LOG
# done


# # python main.py   --cfg configs/source-dist/source-dist-dag-dirgcn-s2gnn-undir.yaml   out_dir tests/results/source-dist-default   wandb.use False   seed 1  # we use 1..10
# for seed in {11..20}
# do
#   echo "Running distance regression(dag-dirgcn-s2gnn-undir ) with seed $seed" | tee -a $LOG
#   START=$(date +%s)
#   python main.py --cfg configs/source-dist/source-dist-dag-dirgcn-s2gnn-undir.yaml \
#     out_dir tests/results/source-dist-dag-dirgcn-s2gnn-undir wandb.use False seed $seed
#   END=$(date +%s)
#   echo "Time: $((END - START)) seconds" | tee -a $LOG
#   echo "---------------------------------------" >> $LOG
# done

# for seed in {11..14}
# do
#   echo "Running distance regression(dag-dirgcn-s2gnn-dir ) with seed $seed" | tee -a $LOG
#   START=$(date +%s)
#   python main.py --cfg configs/source-dist/source-dist-dag-dirgcn-s2gnn-dir.yaml \
#     out_dir tests/results/source-dist-dag-dirgcn-s2gnn-dir wandb.use False seed $seed
#   END=$(date +%s)
#   echo "Time: $((END - START)) seconds" | tee -a $LOG
#   echo "---------------------------------------" >> $LOG
# done

# for seed in {11..20}
# do
#   echo "Over-squashing(spec) with seed $seed" | tee -a $LOG
#   START=$(date +%s)
#   python main.py --cfg configs/over-squashing/over-squashing-spec.yaml \
#     out_dir tests/results/over-squashing-spec wandb.use False seed $seed
#   END=$(date +%s)
#   echo "Time: $((END - START)) seconds" | tee -a $LOG
#   echo "---------------------------------------" >> $LOG
# done

# for seed in {11..20}
# do
#   echo "Over-squashing(gated) with seed $seed" | tee -a $LOG
#   START=$(date +%s)
#   python main.py --cfg configs/over-squashing/over-squashing-gatedgcnconv.yaml \
#     out_dir tests/results/over-squashing-gatedgcnconv wandb.use False seed $seed
#   END=$(date +%s)
#   echo "Time: $((END - START)) seconds" | tee -a $LOG
#   echo "---------------------------------------" >> $LOG
# done

# for seed in {11..14}
# do
#   echo "arxiv-year(s2gnn) with seed $seed" | tee -a $LOG
#   START=$(date +%s)
#   python main.py --cfg configs/arxiv-year/arxiv-year-dirgcn-s2gnn.yaml \
#     out_dir tests/results/arxiv-year/s2gnn wandb.use False seed $seed
#   END=$(date +%s)
#   echo "Time: $((END - START)) seconds" | tee -a $LOG
#   echo "---------------------------------------" >> $LOG
# done

# for seed in {12..13}
# do
#   echo "arxiv-year(s2gnn-pe) with seed $seed" | tee -a $LOG
#   START=$(date +%s)
#   python main.py --cfg configs/arxiv-year/arxiv-year-dirgcn-s2gnn-pe.yaml \
#     out_dir tests/results/arxiv-year/s2gnn-pe wandb.use False seed $seed
#   END=$(date +%s)
#   echo "Time: $((END - START)) seconds" | tee -a $LOG
#   echo "---------------------------------------" >> $LOG
# done

# for seed in {11..13}
# do
#   echo "arxiv-year(lingnn) with seed $seed" | tee -a $LOG
#   START=$(date +%s)
#   python main.py --cfg configs/arxiv-year/arxiv-year-dirgcn-lingnn.yaml \
#     out_dir tests/results/arxiv-year/lingnn wandb.use False seed $seed
#   END=$(date +%s)
#   echo "Time: $((END - START)) seconds" | tee -a $LOG
#   echo "---------------------------------------" >> $LOG
# done

# for seed in {11..13}
# do
#   echo "associative-recall(s2gnn) with seed $seed" | tee -a $LOG
#   START=$(date +%s)
#   python main.py --cfg configs/associative-recall/associative-recall-s2gnn.yaml \
#     out_dir tests/results/associative-recall/s2gnn wandb.use False seed $seed
#   END=$(date +%s)
#   echo "Time: $((END - START)) seconds" | tee -a $LOG
#   echo "---------------------------------------" >> $LOG
# done

# for seed in {11..13}
# do
#   echo "associative-recall(spec) with seed $seed" | tee -a $LOG
#   START=$(date +%s)
#   python main.py --cfg configs/associative-recall/associative-recall-spec.yaml \
#     out_dir tests/results/associative-recall/spec wandb.use False seed $seed
#   END=$(date +%s)
#   echo "Time: $((END - START)) seconds" | tee -a $LOG
#   echo "---------------------------------------" >> $LOG
# done

# for seed in {11..13}
# do
#   echo "associative-recall(lingnn) with seed $seed" | tee -a $LOG
#   START=$(date +%s)
#   python main.py --cfg configs/associative-recall/associative-recall-lingnn.yaml \
#     out_dir tests/results/associative-recall/lingnn wandb.use False seed $seed
#   END=$(date +%s)
#   echo "Time: $((END - START)) seconds" | tee -a $LOG
#   echo "---------------------------------------" >> $LOG
# done


####cluster
# for seed in {11..13}
# do
#   echo "cluster-gcn(100k) with seed $seed" | tee -a $LOG
#   START=$(date +%s)
#   python main.py --cfg configs/cluster/cluster-gcn-100k.yaml \
#     out_dir tests/results/cluster/gcn-100k wandb.use False seed $seed
#   END=$(date +%s)
#   echo "Time: $((END - START)) seconds" | tee -a $LOG
#   echo "---------------------------------------" >> $LOG
# done

# for seed in {11..13}
# do
#   echo "cluster-gcn(s2gnn-100k) with seed $seed" | tee -a $LOG
#   START=$(date +%s)
#   python main.py --cfg configs/cluster/cluster-gcn-s2gnn-100k.yaml \
#     out_dir tests/results/cluster/gcn-s2gnn-100k wandb.use False seed $seed
#   END=$(date +%s)
#   echo "Time: $((END - START)) seconds" | tee -a $LOG
#   echo "---------------------------------------" >> $LOG
# done

# for seed in {11..13}
# do
#   echo "cluster-gcn(s2gnn-pe-100k) with seed $seed" | tee -a $LOG
#   START=$(date +%s)
#   python main.py --cfg configs/cluster/cluster-gcn-s2gnn-pe-100k.yaml \
#     out_dir tests/results/cluster/gcn-s2gnn-pe-100k wandb.use False seed $seed
#   END=$(date +%s)
#   echo "Time: $((END - START)) seconds" | tee -a $LOG
#   echo "---------------------------------------" >> $LOG
# done


# for seed in {11..13}
# do
#   echo "cluster-gat(100k) with seed $seed" | tee -a $LOG
#   START=$(date +%s)
#   python main.py --cfg configs/cluster/cluster-gat-100k.yaml \
#     out_dir tests/results/cluster/gat-100k wandb.use False seed $seed
#   END=$(date +%s)
#   echo "Time: $((END - START)) seconds" | tee -a $LOG
#   echo "---------------------------------------" >> $LOG
# done


# #这个没跑完，下次从这里开始
# for seed in {11..13}
# do
#   echo "cluster-gat(s2gnn-100k) with seed $seed" | tee -a $LOG
#   START=$(date +%s)
#   python main.py --cfg configs/cluster/cluster-gat-s2gnn-100k.yaml \
#     out_dir tests/results/cluster/gat-s2gnn-100k wandb.use False seed $seed
#   END=$(date +%s)
#   echo "Time: $((END - START)) seconds" | tee -a $LOG
#   echo "---------------------------------------" >> $LOG
# done

# for seed in {13}
# do
#   echo "cluster-gat(s2gnn-pe-100k) with seed $seed" | tee -a $LOG
#   START=$(date +%s)
#   python main.py --cfg configs/cluster/cluster-gat-s2gnn-pe-100k.yaml \
#     out_dir tests/results/cluster/gat-s2gnn-pe-100k wandb.use False seed $seed
#   END=$(date +%s)
#   echo "Time: $((END - START)) seconds" | tee -a $LOG
#   echo "---------------------------------------" >> $LOG
# done

# for seed in {11..13}
# do
#   echo "cluster-gatedgcn(100k) with seed $seed" | tee -a $LOG
#   START=$(date +%s)
#   python main.py --cfg configs/cluster/cluster-gatedgcn-100k.yaml \
#     out_dir tests/results/cluster/gatedgcn-100k wandb.use False seed $seed
#   END=$(date +%s)
#   echo "Time: $((END - START)) seconds" | tee -a $LOG
#   echo "---------------------------------------" >> $LOG
# done

# for seed in {11..13}
# do
#   echo "cluster-gatedgcn-pe(100k) with seed $seed" | tee -a $LOG
#   START=$(date +%s)
#   python main.py --cfg configs/cluster/cluster-gatedgcn-pe-100k.yaml \
#     out_dir tests/results/cluster/gatedgcn-pe-100k wandb.use False seed $seed
#   END=$(date +%s)
#   echo "Time: $((END - START)) seconds" | tee -a $LOG
#   echo "---------------------------------------" >> $LOG
# done

# for seed in {11..13}
# do
#   echo "cluster-gatedgcn(s2gnn-100k) with seed $seed" | tee -a $LOG
#   START=$(date +%s)
#   python main.py --cfg configs/cluster/cluster-gatedgcn-s2gnn-100k.yaml \
#     out_dir tests/results/cluster/gatedgcn-s2gnn-100k wandb.use False seed $seed
#   END=$(date +%s)
#   echo "Time: $((END - START)) seconds" | tee -a $LOG
#   echo "---------------------------------------" >> $LOG
# done

# for seed in {11..13}
# do
#   echo "cluster-gatedgcn(s2gnn-pe-100k) with seed $seed" | tee -a $LOG
#   START=$(date +%s)
#   python main.py --cfg configs/cluster/cluster-gatedgcn-s2gnn-pe-100k.yaml \
#     out_dir tests/results/cluster/gatedgcn-s2gnn-pe-100k wandb.use False seed $seed
#   END=$(date +%s)
#   echo "Time: $((END - START)) seconds" | tee -a $LOG
#   echo "---------------------------------------" >> $LOG
# done

#cluster-gmm

# for seed in {11..13}
# do
#   echo "cluster-gmm(gcnconv) with seed $seed" | tee -a $LOG
#   START=$(date +%s)
#   python main.py --cfg configs/custom-cluster/custom-cluster-gmm-gcnconv.yaml \
#     out_dir tests/results/custom-cluster/gmm-gcnconv wandb.use False seed $seed
#   END=$(date +%s)
#   echo "Time: $((END - START)) seconds" | tee -a $LOG
#   echo "---------------------------------------" >> $LOG
# done

# for seed in {12..13}
# do
#   echo "cluster-gmm(gcnconv-pe) with seed $seed" | tee -a $LOG
#   START=$(date +%s)
#   python main.py --cfg configs/custom-cluster/custom-cluster-gmm-gcnconv-pe.yaml \
#     out_dir tests/results/custom-cluster/gmm-gcnconv-pe wandb.use False seed $seed
#   END=$(date +%s)
#   echo "Time: $((END - START)) seconds" | tee -a $LOG
#   echo "---------------------------------------" >> $LOG
# done

# for seed in {11..13}
# do
#   echo "cluster-gmm(s2gnn) with seed $seed" | tee -a $LOG
#   START=$(date +%s)
#   python main.py --cfg configs/custom-cluster/custom-cluster-gmm-s2gnn.yaml \
#     out_dir tests/results/custom-cluster/gmm-s2gnn wandb.use False seed $seed
#   END=$(date +%s)
#   echo "Time: $((END - START)) seconds" | tee -a $LOG
#   echo "---------------------------------------" >> $LOG
# done

# for seed in {11..13}
# do
#   echo "cluster-gmm(s2gnn-pe) with seed $seed" | tee -a $LOG
#   START=$(date +%s)
#   python main.py --cfg configs/custom-cluster/custom-cluster-gmm-s2gnn-pe.yaml \
#     out_dir tests/results/custom-cluster/gmm-s2gnn-pe wandb.use False seed $seed
#   END=$(date +%s)
#   echo "Time: $((END - START)) seconds" | tee -a $LOG
#   echo "---------------------------------------" >> $LOG
# done


for seed in {11..13}
do
  echo "Running distance regression(dag-dirgcn-pe ) with seed $seed" | tee -a $LOG
  START=$(date +%s)
  python main.py --cfg configs/source-dist/source-dist-dag-dirgcn-pe.yaml \
    out_dir tests/results/source-dist/dag-dirgcn-pe wandb.use False seed $seed
  END=$(date +%s)
  echo "Time: $((END - START)) seconds" | tee -a $LOG
  echo "---------------------------------------" >> $LOG
done

for seed in {11..13}
do
  echo "Running distance regression(dag-dirgcn-s2gnn-dir-pe ) with seed $seed" | tee -a $LOG
  START=$(date +%s)
  python main.py --cfg configs/source-dist/source-dist-dag-dirgcn-s2gnn-dir-pe.yaml \
    out_dir tests/results/source-dist/dag-dirgcn-s2gnn-dir-pe wandb.use False seed $seed
  END=$(date +%s)
  echo "Time: $((END - START)) seconds" | tee -a $LOG
  echo "---------------------------------------" >> $LOG
done

for seed in {11..13}
do
  echo "Running distance regression(dag-dirgcn-s2gnn-undir-pe ) with seed $seed" | tee -a $LOG
  START=$(date +%s)
  python main.py --cfg configs/source-dist/source-dist-dag-dirgcn-s2gnn-undir-pe.yaml \
    out_dir tests/results/source-dist/dag-dirgcn-s2gnn-undir-pe wandb.use False seed $seed
  END=$(date +%s)
  echo "Time: $((END - START)) seconds" | tee -a $LOG
  echo "---------------------------------------" >> $LOG
done