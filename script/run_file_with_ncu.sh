dirc=$3
name=$1
out=$2
# ncu ${dirc}${name} | grep "Duration" | awk -v t="${name}" '{print $2t,$3,$2}' OFS=","
ncu --clock-control none ${dirc}${name} | grep "Duration" | awk -v t="${name}" '{printf "%s,%s,%s\n", $4t,$3,$2}' >> ${out}