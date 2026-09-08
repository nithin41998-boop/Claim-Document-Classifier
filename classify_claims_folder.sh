#!/bin/bash
###############################################################################
# classify_claims_folder.sh
#
# Standalone claim-document triage tool. Separate from the identity-document
# dimension-check pipeline. Classifies every file in an input folder into one
# of Truuth's established claim-document categories (categories.txt), groups
# files that appear to be multiple pages/sides of the same physical document
# (e.g. front and back of a licence), and organises copies into per-category
# output folders alongside a summary CSV.
#
# Usage:
#   ./classify_claims_folder.sh -d <input_folder> -o <output_folder>
###############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NOCOLOUR='\033[0m'

usage="Usage: $0 -d <input_folder> -o <output_folder>"

inputFolder=""
outputFolder=""

while getopts ":d:o:" opt; do
	case $opt in
		d)
			inputFolder="$OPTARG"
			;;
		o)
			outputFolder="$OPTARG"
			;;
		\?)
			echo "Invalid switch."
			echo -e "$usage"
			exit 1
			;;
	esac
done

if [ -z "$inputFolder" ] || [ -z "$outputFolder" ]; then
	echo "Missing required switch."
	echo -e "$usage"
	exit 1
fi

if [ ! -d "$inputFolder" ]; then
	echo -e "${RED}Input folder not found:${NOCOLOUR} $inputFolder"
	exit 1
fi

if [ ! -f "$SCRIPT_DIR/categories.txt" ]; then
	echo -e "${RED}categories.txt not found in $SCRIPT_DIR${NOCOLOUR}"
	exit 1
fi

mkdir -p "$outputFolder"
export TOKEN_LOG_FILE="$outputFolder/token_usage.log"
touch "$TOKEN_LOG_FILE"

summaryFile="$outputFolder/summary.csv"
echo "Original Filename,Category,Group ID" > "$summaryFile"

groupKeyFor() {
	local fname="$1"
	local base="${fname%.*}"
	base=$(echo "$base" | sed -E 's/[-_ ]?(front|back|page[0-9]+|p[0-9]+|[0-9]+)$//I')
	echo "$base"
}

sanitizeFolderName() {
	local name="$1"
	echo "$name" | sed -E 's#[/\\:*?"<>|]#-#g' | sed -E 's/[[:space:]]+$//'
}

getClassifiableImage() {
	local filePath="$1"
	local extension="${filePath##*.}"
	if [[ "$extension" =~ ^(pdf|PDF)$ ]]; then
		local tmpRendered="/tmp/claim_classify_$$"
		pdftoppm -jpeg -r 150 -f 1 -l 1 "$filePath" "$tmpRendered" 2>/dev/null
		local rendered
		rendered=$(ls "${tmpRendered}"* 2>/dev/null | head -1)
		echo "$rendered"
	else
		echo "$filePath"
	fi
}

fileCount=0
totalFiles=$(find "$inputFolder" -maxdepth 1 -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.pdf" \) | wc -l)

declare -A groupCounter

while IFS= read -r -d '' filePath; do
	fileCount=$((fileCount + 1))
	fileName=$(basename "$filePath")
	echo -e "${GREEN}[$fileCount/$totalFiles]${NOCOLOUR} Classifying: $fileName"

	classifyImg=$(getClassifiableImage "$filePath")
	tmpWasRendered=0
	[[ "$classifyImg" != "$filePath" ]] && tmpWasRendered=1

	if [ -z "$classifyImg" ] || [ ! -f "$classifyImg" ]; then
		category="Unclassified"
	else
		category=$(python3 "$SCRIPT_DIR/classify_claim_document.py" "$classifyImg" 2>/dev/null)
		[ -z "$category" ] && category="Unclassified"
	fi

	if [ "$tmpWasRendered" -eq 1 ]; then
		rm -f "$classifyImg"
	fi

	groupKey="${category}::$(groupKeyFor "$fileName")"
	if [ -z "${groupCounter[$groupKey]}" ]; then
		groupCounter[$groupKey]=$(echo -n "$groupKey" | md5sum | cut -c1-8)
	fi
	groupId="${groupCounter[$groupKey]}"

	echo "\"$fileName\",\"$category\",\"$groupId\"" >> "$summaryFile"

	safeCategory=$(sanitizeFolderName "$category")
	categoryFolder="$outputFolder/$safeCategory"
	mkdir -p "$categoryFolder"

	echo "$fileName|$groupId" >> "$outputFolder/.grouping_tmp"

done < <(find "$inputFolder" -maxdepth 1 -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.pdf" \) -print0)

if [ -f "$outputFolder/.grouping_tmp" ]; then
	sort -t'|' -k2 "$outputFolder/.grouping_tmp" | awk -F'|' '{print $2}' | sort | uniq -c | while read -r count gid; do
		echo "$count $gid" >> "$outputFolder/.groupcounts_tmp"
	done

	while IFS='|' read -r fname gid; do
		category=$(grep -F "\"$fname\"," "$summaryFile" | head -1 | awk -F'","' '{print $2}')
		safeCategory=$(sanitizeFolderName "$category")
		categoryFolder="$outputFolder/$safeCategory"
		count=$(grep " $gid$" "$outputFolder/.groupcounts_tmp" 2>/dev/null | awk '{print $1}')
		srcPath="$inputFolder/$fname"
		if [ "$count" -gt 1 ] 2>/dev/null; then
			mkdir -p "$categoryFolder/$gid"
			cp "$srcPath" "$categoryFolder/$gid/"
		else
			cp "$srcPath" "$categoryFolder/"
		fi
	done < "$outputFolder/.grouping_tmp"
	rm -f "$outputFolder/.grouping_tmp" "$outputFolder/.groupcounts_tmp"
fi

echo ""
echo -e "${GREEN}###### Claim Document Classification Summary ######${NOCOLOUR}"
echo "Total files processed: $fileCount"
echo "Summary CSV: $summaryFile"
echo "Categorised documents: $outputFolder/<Category Name>/"

if [ -s "$TOKEN_LOG_FILE" ]; then
	totalInputTokens=$(awk -F, '{sum+=$1} END {print sum+0}' "$TOKEN_LOG_FILE")
	totalOutputTokens=$(awk -F, '{sum+=$2} END {print sum+0}' "$TOKEN_LOG_FILE")
	echo ""
	echo -e "${GREEN}###### LLM Token Usage Summary ######${NOCOLOUR}"
	echo "Total input tokens: $totalInputTokens"
	echo "Total output tokens: $totalOutputTokens"
fi
