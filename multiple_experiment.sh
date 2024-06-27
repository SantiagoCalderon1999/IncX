# CPUs limit is roughly 380 (Estimation)

for (( j=1; j<=380; j++ ))
do
    sbatch batch_experiment.sh
done

