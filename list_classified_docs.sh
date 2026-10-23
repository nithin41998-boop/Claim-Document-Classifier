#!/bin/bash
outputFolder="$1"
summaryFile="$outputFolder/summary.csv"

if [ -z "$outputFolder" ]; then
    echo "Usage: $0 <output-folder>"
    exit 1
fi

if [ ! -f "$summaryFile" ]; then
    echo "summary.csv not found at: $summaryFile"
    exit 1
fi

totalCount=$(($(wc -l < "$summaryFile") - 1))
echo "###### Documents Handled by Classifier ######"
echo "Total documents: $totalCount"
echo ""

tail -n +2 "$summaryFile" | awk -F'","' '{
    fname = $1; gsub(/^"/, "", fname);
    category = $2;
    print category "|" fname
}' | sort -t'|' -k1,1 -k2,2 | awk -F'|' '
{
    if ($1 != prevCat) {
        if (prevCat != "") print "";
        print "== " $1 " ==";
        prevCat = $1;
    }
    print "  - " $2;
}'
