#!/usr/bin/env perl
#
# This file is part of moses.  Its use is licensed under the GNU Lesser General
# Public License version 2.1 or, at your option, any later version.

use warnings;
use strict;
use utf8;
use Encode;
use XML::Twig;

binmode STDOUT, ":utf8";
binmode STDERR, ":utf8";


#################################
# History:
#
# version 13a
#    * modified the scoring functions to prevent division-by-zero errors when a system segment is empty
#        * affected methods: 'bleu_score' and 'bleu_score_smoothing'
#
# version 13
#    * Uses a XML parser to read data (only when extension is .xml)
#    * Smoothing of the segment-level BLEU scores, done by default
#        * smoothing method similar to that of bleu-1.04.pl (IBM)
#        * see comments above the 'bleu_score' method for more details on how the smoothing is computed
#        * added a '--no-smoothing' option to simulate old scripts behavior
#    * Introduction of the 'brevity-penalty' option, taking one of two values:
#        * 'closest' (default) : act as IBM BLEU (taking the closest reference translation length)
#            * in case two reference translations are at the same distance, will take the shortest one
#            * for more details regarding how the BP is computed, see comments of the 'brevity_penalty_closest' function
#        * 'shortest' : act as previous versions of the script (taking shortest reference translation length)
#    * Introduction of the 'international-tokenization' option, boolean, disabled by default
#        by default (when the option is not provided), uses 11b's tokenization function
#        when option specified, uses v12's tokenization function
#    * Introduction of a 'Metrics MATR output' flag (option '--metricsMATR')
#        when used, creates three files for both BLEU score and NIST score:
#            * BLEU-seg.scr and NIST-seg.scr: contain segment-level scores
#            * BLEU-doc.scr and NIST-doc.scr: contain document-level scores
#            * BLEU-sys.scr and NIST-sys.scr: contain system-level scores
#    * SGML parsing
#        * script will halt if source, reference and test files don't share the same setid attribute value (used for metricsMATR output)
#        * correct segment IDs extracted from the files (was previously using an array, and using the index as a segID for output)
#    * detailed output flag (-d) can now be used when running both BLEU and NIST
#
# version 12
#    * Text normalization changes:
#        * convert entity references (only the entities declared in the DTD)
#        * now uses unicode categories
#        * tokenize punctuation unless followed AND preceded by digits
#        * tokenize symbols
#    * UTF-8 handling:
#        * files are now read using utf8 mode
#    * Added the '-e' command-line option to enclose non-ASCII characters between spaces
#
# version 11b -- text normalization modified:
#    * take out the join digit line because it joins digits
#      when it shouldn't have
#      $norm_text =~ s/(\d)\s+(?=\d)/$1/g; #join digits
#
# version 11a -- corrected output of individual n-gram precision values
#
# version 11 -- bug fixes:
#    * make filehandle operate in binary mode to prevent Perl from operating
#      (by default in Red Hat 9) in UTF-8
#    * fix failure on joining digits
# version 10 -- updated output to include more details of n-gram scoring.
#    Defaults to generate both NIST and BLEU scores.  Use -b for BLEU
#    only, use -n for NIST only
#
# version 09d -- bug fix (for BLEU scoring, ngrams were fixed at 4
#    being the max, regardless what was entered on the command line.)
#
# version 09c -- bug fix (During the calculation of ngram information,
#    each ngram was being counted only once for each segment.  This has
#    been fixed so that each ngram is counted correctly in each segment.)
#
# version 09b -- text normalization modified:
#    * option flag added to preserve upper case
#    * non-ASCII characters left in place.
#
# version 09a -- text normalization modified:
#    * &quot; and &amp; converted to "" and &, respectively
#    * non-ASCII characters kept together (bug fix)
#
# version 09 -- modified to accommodate sgml tag and attribute
#    names revised to conform to default SGML conventions.
#
# version 08 -- modifies the NIST metric in accordance with the
#    findings on the 2001 Chinese-English dry run corpus.  Also
#    incorporates the BLEU metric as an option and supports the
#    output of ngram detail.
#
# version 07 -- in response to the MT meeting on 28 Jan 2002 at ISI
#    Keep strings of non-ASCII characters together as one word
#    (rather than splitting them into one-character words).
#    Change length penalty so that translations that are longer than
#    the average reference translation are not penalized.
#
# version 06
#    Prevent divide-by-zero when a segment has no evaluation N-grams.
#    Correct segment index for level 3 debug output.
#
# version 05
#    improve diagnostic error messages
#
# version 04
#    tag segments
#
# version 03
#    add detailed output option (intermediate document and segment scores)
#
# version 02
#    accommodation of modified sgml tags and attributes
#
# version 01
#    same as bleu version 15, but modified to provide formal score output.
#
# original IBM version
#    Author: Kishore Papineni
#    Date: 06/10/2001
#################################

######
# Intro
my ($date, $time) = date_time_stamp();
print "MT evaluation scorer began on $date at $time\n";
print "command line:  ", $0, " ", join(" ", @ARGV), "\n";
my $usage = "\n\nUsage: $0 -r <ref_file> -s <src_file> -t <tst_file>\n\n".
    "Description:  This Perl script evaluates MT system performance.\n".
    "\n".
    "Required arguments:\n".
    "  -r <ref_file> is a file containing the reference translations for\n".
    "      the documents to be evaluated.\n".
    "  -s <src_file> is a file containing the source documents for which\n".
    "      translations are to be evaluated\n".
    "  -t <tst_file> is a file containing the translations to be evaluated\n".
    "\n".
    "Optional arguments:\n".
    "  -h prints this help message to STDOUT\n".
    "  -c preserves upper-case alphabetic characters\n".
    "  -b generate BLEU scores only\n".
    "  -n generate NIST scores only\n".
    "  -d detailed output flag:\n".
    "         0 (default) for system-level score only\n".
    "         1 to include document-level scores\n".
    "         2 to include segment-level scores\n".
    "         3 to include ngram-level scores\n".
    "  -e enclose non-ASCII characters between spaces\n".
     "  --brevity-penalty ( closest | shortest )\n" .
    "         closest (default) : acts as IBM BLEU (takes the closest reference translation length)\n" .
    "         shortest : acts as previous versions of the script (takes the shortest reference translation length)\n" .
    "  --international-tokenization\n" .
    "         when specified, uses Unicode-based (only) tokenization rules\n" .
    "         when not specified (default), uses default tokenization (some language-dependant rules)\n" .
    "  --metricsMATR : create three files for both BLEU scores and NIST scores:\n" .
    "         BLEU-seg.scr and NIST-seg.scr : segment-level scores\n" .
    "         BLEU-doc.scr and NIST-doc.scr : document-level scores\n" .
    "         BLEU-sys.scr and NIST-sys.scr : system-level scores\n" .
    "  --no-smoothing : disable smoothing on BLEU scores\n" .
    "\n";

use vars qw ($opt_r $opt_s $opt_t $opt_d $opt_h $opt_b $opt_n $opt_c $opt_x $opt_e);
use Getopt::Long;
my $ref_file = '';
my $src_file = '';
my $tst_file = '';
my $detail = 0;
my $help = '';
my $preserve_case = '';
my $split_non_ASCII = '';
my $brevity_penalty = 'closest';
my $international_tokenization;
my $metricsMATR_output = '';
my $no_smoothing = '';
our $opt_x = '';
our $opt_b = '';
our $opt_n = '';
GetOptions(
	'r=s' => \$ref_file,
	's=s' => \$src_file,
	't=s' => \$tst_file,
	'd:i' => \$detail,
	'h|help' => \$help,
	'b',
	'n',
	'c' => \$preserve_case,
	'x:s',
	'e' => \$split_non_ASCII,
	'brevity-penalty:s' => \$brevity_penalty,
	'international-tokenization' => \$international_tokenization,
	'metricsMATR-output' => \$metricsMATR_output,
	'no-smoothing' => \$no_smoothing
);
die $usage if $help;

die "Error in command line:  ref_file not defined$usage" unless ( $ref_file );
die "Error in command line:  src_file not defined$usage" unless ( $src_file );
die "Error in command line:  tst_file not defined$usage" unless ( $tst_file );
my $BLEU_BP;
if ( !( $brevity_penalty cmp 'closest' ) )
{
	$BLEU_BP = \&brevity_penalty_closest;
}
elsif ( !( $brevity_penalty cmp 'shortest' ) )
{
	$BLEU_BP = \&brevity_penalty_shortest;
}
else
{
	die "Incorrect value supplied for 'brevity_penalty'$usage";
}
my $TOKENIZATION = \&tokenization;
$TOKENIZATION = \&tokenization_international if ( $international_tokenization );

my $BLEU_SCORE = \&bleu_score;
$BLEU_SCORE = \&bleu_score_nosmoothing if ( $no_smoothing );

my $max_Ngram = 9;

my $METHOD = "BOTH";
if ( $opt_b ) { $METHOD = "BLEU"; }
if ( $opt_n ) { $METHOD = "NIST"; }
my $method;

######
# Global variables
my ($src_lang, $tgt_lang, @tst_sys, @ref_sys); # evaluation parameters
my (%tst_data, %ref_data); # the data -- with structure:  {system}{document}{segments}
my ($src_id, $ref_id, $tst_id); # unique identifiers for ref and tst translation sets
my %eval_docs;     # document information for the evaluation data set
my %ngram_info;    # the information obtained from (the last word in) the ngram

######
# Get source document ID's
($src_id) = get_source_info ($src_file);

######
# Get reference translations
($ref_id) = get_MT_data (\%ref_data, "RefSet", $ref_file);

compute_ngram_info ();

######
# Get translations to evaluate
($tst_id) = get_MT_data (\%tst_data, "TstSet", $tst_file);

######
# Check data for completeness and correctness
check_MT_data ();

######
#
my %NISTmt;
my %NISTOverall;
my %BLEUmt;
my %BLEUOverall;

######
# Evaluate
print "  Evaluation of $src_lang-to-$tgt_lang translation using:\n";
my $cum_seg = 0;
foreach my $doc (sort keys %eval_docs)
{
	$cum_seg += scalar( keys( %{$eval_docs{$doc}{SEGS}} ) );
}
print "    src set \"$src_id\" (", scalar keys %eval_docs, " docs, $cum_seg segs)\n";
print "    ref set \"$ref_id\" (", scalar keys %ref_data, " refs)\n";
print "    tst set \"$tst_id\" (", scalar keys %tst_data, " systems)\n\n";

foreach my $sys (sort @tst_sys)
{
	for (my $n=1; $n<=$max_Ngram; $n++)
	{
		$NISTmt{$n}{$sys}{cum} = 0;
		$NISTmt{$n}{$sys}{ind} = 0;
		$BLEUmt{$n}{$sys}{cum} = 0;
		$BLEUmt{$n}{$sys}{ind} = 0;
	}
	if ( ($METHOD eq "BOTH") || ($METHOD eq "NIST") )
	{
		$method="NIST";
		score_system ($sys, \%NISTmt, \%NISTOverall);
	}
	if ( ($METHOD eq "BOTH") || ($METHOD eq "BLEU") )
	{
		$method="BLEU";
		score_system ($sys, \%BLEUmt, \%BLEUOverall);
	}
}

######
printout_report ();
if ( $metricsMATR_output )
{
	outputMetricsMATR( 'NIST', %NISTOverall ) if ( ( $METHOD eq 'BOTH' ) || ( $METHOD eq 'NIST' ) );
	outputMetricsMATR( 'BLEU', %BLEUOverall ) if ( ( $METHOD eq 'BOTH' ) || ( $METHOD eq 'BLEU' ) );
}

($date, $time) = date_time_stamp();
print "MT evaluation scorer ended on $date at $time\n";

exit 0;

#################################

sub get_source_info
{
	my ($file) = @_;
	my ($name, $id, $src, $doc, $seg);
	my ($data, $tag, $span);

	# Extension of the file determines the parser used:
	#   .xml       : XML::Twig
	#   otherwise  : simple SGML parsing functions
	if ( $file =~ /\.xml$/i )
	{
		my $twig = XML::Twig->new();
		$twig->parsefile( $file );
		my $root = $twig->root;
		my $currentSet = $root->first_child( 'srcset' );
		die "Source XML file '$file' does not contain the 'srcset' element" if ( not $currentSet );
		$id = $currentSet->{ 'att' }->{ 'setid' } or die "No 'setid' attribute value in '$file'";
		$src = $currentSet->{ 'att' }->{ 'srclang' } or die "No srcset 'srclang' attribute value in '$file'";
		die "Not the same srclang attribute values across sets" unless ( not defined $src_lang or $src eq $src_lang );
		$src_lang = $src;
		foreach my $currentDoc ( $currentSet->get_xpath( './/doc' ) )
		{
			my $docID = $currentDoc->{ 'att' }->{ 'docid' } or die "No document 'docid' attribute value in '$file'";
			foreach my $currentSeg ( $currentDoc->get_xpath( './/seg' ) )
			{
				my $segID = $currentSeg->{ 'att' }->{ 'id' } or die "No segment 'id' attribute value in '$file'";
				my $segData = $currentSeg->text;
				($eval_docs{$docID}{SEGS}{$segID}) = &{ $TOKENIZATION }( $segData );
			}
		}
	}
	else
	{
		#read data from file
		open (FILE, $file) or die "\nUnable to open translation data file '$file'", $usage;
		binmode FILE, ":utf8";
		$data .= $_ while <FILE>;
		close (FILE);

		#get source set info
		die "\n\nFATAL INPUT ERROR:  no 'src_set' tag in src_file '$file'\n\n"
			unless ($tag, $span, $data) = extract_sgml_tag_and_span ("SrcSet", $data);
		die "\n\nFATAL INPUT ERROR:  no tag attribute '$name' in file '$file'\n\n"
			unless ($id) = extract_sgml_tag_attribute ($name="SetID", $tag);
		die "\n\nFATAL INPUT ERROR:  no tag attribute '$name' in file '$file'\n\n"
			unless ($src) = extract_sgml_tag_attribute ($name="SrcLang", $tag);
		die "\n\nFATAL INPUT ERROR:  $name ('$src') in file '$file' inconsistent\n"
			."                    with $name in previous input data ('$src_lang')\n\n"
			unless (not defined $src_lang or $src eq $src_lang);
		$src_lang = $src;

		#get doc info -- ID and # of segs
		$data = $span;
		while (($tag, $span, $data) = extract_sgml_tag_and_span ("Doc", $data))
		{
			die "\n\nFATAL INPUT ERROR:  no tag attribute '$name' in file '$file'\n\n"
				unless ($doc) = extract_sgml_tag_attribute ($name="DocID", $tag);
			die "\n\nFATAL INPUT ERROR:  duplicate '$name' in file '$file'\n\n"
				if defined $eval_docs{$doc};
			$span =~ s/[\s\n\r]+/ /g;  # concatenate records
			my $nseg=0, my $seg_data = $span;
			while (($tag, $span, $seg_data) = extract_sgml_tag_and_span ("Seg", $seg_data))
			{
				die "\n\nFATAL INPUT ERROR:  no attribute '$name' in file '$file'\n\n"
					unless ($seg) = extract_sgml_tag_attribute( $name='id', $tag );
				($eval_docs{$doc}{SEGS}{$seg}) = &{ $TOKENIZATION }( $span );
				$nseg++;
			}
			die "\n\nFATAL INPUT ERROR:  no segments in document '$doc' in file '$file'\n\n"
				if $nseg == 0;
		}
		die "\n\nFATAL INPUT ERROR:  no documents in file '$file'\n\n"
			unless keys %eval_docs > 0;
	}
	return $id;
}

#################################

sub get_MT_data
{
	my ($docs, $set_tag, $file) = @_;
	my ($name, $id, $src, $tgt, $sys, $doc, $seg);
	my ($tag, $span, $data);

	# Extension of the file determines the parser used:
	#   .xml       : XML::Twig
	#   otherwise  : simple SGML parsing functions
	if ( $file =~ /\.xml$/i )
	{
		my $twig = XML::Twig->new();
		$twig->parsefile( $file );
		my $root = $twig->root;
		foreach my $currentSet ( $root->get_xpath( 'refset' ), $root->get_xpath( 'tstset' ) )
		{
			$id = $currentSet->{ 'att' }->{ 'setid' } or die "No 'setid' attribute value in '$file'";
			$src = $currentSet->{ 'att' }->{ 'srclang' } or die "No 'srclang' attribute value in '$file'";
			$tgt = $currentSet->{ 'att' }->{ 'trglang' } or die "No 'trglang' attribute value in '$file'";
			die "Not the same 'srclang' attribute value across sets" unless ( $src eq $src_lang );
			die "Not the same 'trglang' attribute value across sets" unless ( ( not defined $tgt_lang ) or ( $tgt = $tgt_lang ) );
			$tgt_lang = $tgt;
			my $sys;
			if ( $currentSet->name eq 'tstset' )
			{
				$sys = $currentSet->{ 'att' }->{ 'sysid' } or die "No 'sysid' attribute value in '$file'";
			}
			else
			{
				$sys = $currentSet->{ 'att' }->{ 'refid' } or die "No 'refid' attribute value in '$file'";
			}
			foreach my $currentDoc ( $currentSet->get_xpath( './/doc' ) )
			{
				my $docID = $currentDoc->{ 'att' }->{ 'docid' } or die "No document 'docid' attribute value in '$file'";
				$docs->{ $sys }{ $docID }{ FILE } = $file;
				foreach my $currentSeg ( $currentDoc->get_xpath( './/seg' ) )
				{
					my $segID = $currentSeg->{ 'att' }->{ 'id' } or die "No segment 'id' attribute value in '$file'";
					my $segData = $currentSeg->text;
					($docs->{$sys}{$docID}{SEGS}{$segID}) = &{ $TOKENIZATION }( $segData );
				}
			}
		}
	}
	else
	{
		#read data from file
		open (FILE, $file) or die "\nUnable to open translation data file '$file'", $usage;
		binmode FILE, ":utf8";
		$data .= $_ while <FILE>;
		close (FILE);

		#get tag info
		while (($tag, $span, $data) = extract_sgml_tag_and_span ($set_tag, $data))
		{
			die "\n\nFATAL INPUT ERROR:  no tag attribute '$name' in file '$file'\n\n"
				unless ($id) = extract_sgml_tag_attribute ($name="SetID", $tag);
			die "\n\nFATAL INPUT ERROR:  no tag attribute '$name' in file '$file'\n\n"
				unless ($src) = extract_sgml_tag_attribute ($name="SrcLang", $tag);
			die "\n\nFATAL INPUT ERROR:  $name ('$src') in file '$file' inconsistent\n"
				."                    with $name of source ('$src_lang')\n\n"
				unless $src eq $src_lang;
			die "\n\nFATAL INPUT ERROR:  no tag attribute '$name' in file '$file'\n\n"
				unless ($tgt) = extract_sgml_tag_attribute ($name="TrgLang", $tag);
			die "\n\nFATAL INPUT ERROR:  $name ('$tgt') in file '$file' inconsistent\n"
				."                    with $name of the evaluation ('$tgt_lang')\n\n"
				unless (not defined $tgt_lang or $tgt eq $tgt_lang);
			$tgt_lang = $tgt;

			my $mtdata = $span;
			while (($tag, $span, $mtdata) = extract_sgml_tag_and_span ("Doc", $mtdata))
			{
				die "\n\nFATAL INPUT ERROR:  no tag attribute '$name' in file '$file'\n\n"
					unless (my $sys) = extract_sgml_tag_attribute ($name="SysID", $tag);
				die "\n\nFATAL INPUT ERROR:  no tag attribute '$name' in file '$file'\n\n"
					unless $doc = extract_sgml_tag_attribute ($name="DocID", $tag);
				die "\n\nFATAL INPUT ERROR:  document '$doc' for system '$sys' in file '$file'\n"
					."                    previously loaded from file '$docs->{$sys}{$doc}{FILE}'\n\n"
					unless (not defined $docs->{$sys}{$doc});

				$span =~ s/[\s\n\r]+/ /g;  # concatenate records
				my $nseg=0, my $seg_data = $span;
				while (($tag, $span, $seg_data) = extract_sgml_tag_and_span ("Seg", $seg_data))
				{
					die "\n\nFATAIL INPUT ERROR:  no tag attribute '$name' in file '$file'\n\n"
						unless $seg = extract_sgml_tag_attribute( $name="id", $tag );
					($docs->{$sys}{$doc}{SEGS}{$seg}) = &{ $TOKENIZATION }( $span );
					$nseg++;
				}
				die "\n\nFATAL INPUT ERROR:  no segments in document '$doc' in file '$file'\n\n" if $nseg == 0;
				$docs->{$sys}{$doc}{FILE} = $file;
			}
		}
	}
	return $id;
}

#################################

sub check_MT_data
{
	@tst_sys = sort keys %tst_data;
	@ref_sys = sort keys %ref_data;

	die "Not the same 'setid' attribute values across files" unless ( ( $src_id eq $tst_id ) && ( $src_id eq $ref_id ) );

#every evaluation document must be represented for every system and every reference
	foreach my $doc (sort keys %eval_docs)
	{
		my $nseg_source = scalar( keys( %{$eval_docs{$doc}{SEGS}} ) );
		foreach my $sys (@tst_sys)
		{
			die "\n\nFATAL ERROR:  no document '$doc' for system '$sys'\n\n" unless defined $tst_data{$sys}{$doc};
			my $nseg = scalar( keys( %{$tst_data{$sys}{$doc}{SEGS}} ) );
			die "\n\nFATAL ERROR:  translated documents must contain the same # of segments as the source, but\n"
				."              document '$doc' for system '$sys' contains $nseg segments, while\n"
				."              the source document contains $nseg_source segments.\n\n"
				unless $nseg == $nseg_source;
		}
		foreach my $sys (@ref_sys)
		{
			die "\n\nFATAL ERROR:  no document '$doc' for reference '$sys'\n\n" unless defined $ref_data{$sys}{$doc};
			my $nseg = scalar( keys( %{$ref_data{$sys}{$doc}{SEGS}} ) );
			die "\n\nFATAL ERROR:  translated documents must contain the same # of segments as the source, but\n"
				."              document '$doc' for system '$sys' contains $nseg segments, while\n"
				."              the source document contains $nseg_source segments.\n\n"
				unless $nseg == $nseg_source;
		}
	}
}

#################################

sub compute_ngram_info
{
	my ($ref, $doc, $seg);
	my (@wrds, $tot_wrds, %ngrams, $ngram, $mgram);
	my (%ngram_count, @tot_ngrams);

	foreach $ref (keys %ref_data)
	{
		foreach $doc (keys %{$ref_data{$ref}})
		{
			foreach $seg ( keys %{$ref_data{$ref}{$doc}{SEGS}})
			{
				@wrds = split /\s+/, $ref_data{ $ref }{ $doc }{ SEGS }{ $seg };
				$tot_wrds += @wrds;
				%ngrams = %{Words2Ngrams (@wrds)};
				foreach $ngram (keys %ngrams)
				{
					$ngram_count{$ngram} += $ngrams{$ngram};
				}
			}
		}
	}

	foreach $ngram (keys %ngram_count)
	{
		@wrds = split / /, $ngram;
		pop @wrds, $mgram = join " ", @wrds;
		$ngram_info{$ngram} = - log ($mgram ? $ngram_count{$ngram}/$ngram_count{$mgram} : $ngram_count{$ngram}/$tot_wrds) / log 2;
		if (defined $opt_x and $opt_x eq "ngram info")
		{
			@wrds = split / /, $ngram;
			printf "ngram info:%9.4f%6d%6d%8d%3d %s\n", $ngram_info{$ngram}, $ngram_count{$ngram},
				$mgram ? $ngram_count{$mgram} : $tot_wrds, $tot_wrds, scalar @wrds, $ngram;
		}
	}
}

#################################

sub score_system
{
	my ($sys, $ref, $doc, $SCOREmt, $overallScore);
	($sys, $SCOREmt, $overallScore) = @_;
	my ($ref_length, $match_cnt, $tst_cnt, $ref_cnt, $tst_info, $ref_info);
	my ($cum_ref_length, @cum_match, @cum_tst_cnt, @cum_ref_cnt, @cum_tst_info, @cum_ref_info);

	$cum_ref_length = 0;
	for (my $j=1; $j<=$max_Ngram; $j++)
	{
		$cum_match[$j] = $cum_tst_cnt[$j] = $cum_ref_cnt[$j] = $cum_tst_info[$j] = $cum_ref_info[$j] = 0;
	}
	foreach $doc (sort keys %eval_docs)
	{
		($ref_length, $match_cnt, $tst_cnt, $ref_cnt, $tst_info, $ref_info) = score_document ($sys, $doc, $overallScore);
		if ( $method eq "NIST" )
		{
			my %DOCmt = ();
			my $docScore = nist_score( scalar( @ref_sys ), $match_cnt, $tst_cnt, $ref_cnt, $tst_info, $ref_info, $sys, \%DOCmt );
			$overallScore->{ $sys }{ 'documents' }{ $doc }{ 'score' } = $docScore;
			if ( $detail >= 1 )
			{
				printf "$method score using   5-grams = %.4f for system \"$sys\" on document \"$doc\" (%d segments, %d words)\n",
				$docScore, scalar keys %{$tst_data{$sys}{$doc}{SEGS}}, $tst_cnt->[1];
			}
		}

		if ( $method eq "BLEU" )
		{
			my %DOCmt = ();
			my $docScore = &{$BLEU_SCORE}( $ref_length, $match_cnt, $tst_cnt, $sys, \%DOCmt );
			$overallScore->{ $sys }{ 'documents' }{ $doc }{ 'score' } = $docScore;
			if ( $detail >= 1 )
			{
				printf "$method score using   4-grams = %.4f for system \"$sys\" on document \"$doc\" (%d segments, %d words)\n",
					$docScore, scalar keys %{$tst_data{$sys}{$doc}{SEGS}}, $tst_cnt->[1];
			}
		}

		$cum_ref_length += $ref_length;
		for (my $j=1; $j<=$max_Ngram; $j++)
		{
			$cum_match[$j] += $match_cnt->[$j];
			$cum_tst_cnt[$j] += $tst_cnt->[$j];
			$cum_ref_cnt[$j] += $ref_cnt->[$j];
			$cum_tst_info[$j] += $tst_info->[$j];
			$cum_ref_info[$j] += $ref_info->[$j];
			printf "document info: $sys $doc %d-gram %d %d %d %9.4f %9.4f\n", $j, $match_cnt->[$j],
				$tst_cnt->[$j], $ref_cnt->[$j], $tst_info->[$j], $ref_info->[$j]
				if (defined $opt_x and $opt_x eq "document info");
		}
	}

	if ($method eq "BLEU")
	{
		$overallScore->{ $sys }{ 'score' } = &{$BLEU_SCORE}($cum_ref_length, \@cum_match, \@cum_tst_cnt, $sys, $SCOREmt, 1);
	}
	if ($method eq "NIST")
	{
		$overallScore->{ $sys }{ 'score' } = nist_score (scalar @ref_sys, \@cum_match, \@cum_tst_cnt, \@cum_ref_cnt, \@cum_tst_info, \@cum_ref_info, $sys, $SCOREmt);
	}
}

#################################

sub score_document
{
	my ($sys, $ref, $doc, $overallScore);
	($sys, $doc, $overallScore) = @_;
	my ($ref_length, $match_cnt, $tst_cnt, $ref_cnt, $tst_info, $ref_info);
	my ($cum_ref_length, @cum_match, @cum_tst_cnt, @cum_ref_cnt, @cum_tst_info, @cum_ref_info);

	$cum_ref_length = 0;
	for (my $j=1; $j<=$max_Ngram; $j++)
	{
		$cum_match[$j] = $cum_tst_cnt[$j] = $cum_ref_cnt[$j] = $cum_tst_info[$j] = $cum_ref_info[$j] = 0;
	}

#score each segment
	foreach my $seg ( sort{ $a <=> $b } keys( %{$tst_data{$sys}{$doc}{SEGS}} ) )
	{
		my @ref_segments = ();
		foreach $ref (@ref_sys)
		{
			push @ref_segments, $ref_data{$ref}{$doc}{SEGS}{$seg};
			if ( $detail >= 3 )
			{
				printf "ref '$ref', seg $seg: %s\n", $ref_data{$ref}{$doc}{SEGS}{$seg}
			}

		}

		printf "sys '$sys', seg $seg: %s\n", $tst_data{$sys}{$doc}{SEGS}{$seg} if ( $detail >= 3 );
		($ref_length, $match_cnt, $tst_cnt, $ref_cnt, $tst_info, $ref_info) = score_segment ($tst_data{$sys}{$doc}{SEGS}{$seg}, @ref_segments);

		if ( $method eq "BLEU" )
		{
			my %DOCmt = ();
			my $segScore = &{$BLEU_SCORE}($ref_length, $match_cnt, $tst_cnt, $sys, %DOCmt);
			$overallScore->{ $sys }{ 'documents' }{ $doc }{ 'segments' }{ $seg }{ 'score' } = $segScore;
			if ( $detail >= 2 )
			{
				printf "  $method score using 4-grams = %.4f for system \"$sys\" on segment $seg of document \"$doc\" (%d words)\n", $segScore, $tst_cnt->[1]
			}
		}
		if ( $method eq "NIST" )
		{
			my %DOCmt = ();
			my $segScore = nist_score (scalar @ref_sys, $match_cnt, $tst_cnt, $ref_cnt, $tst_info, $ref_info, $sys, %DOCmt);
			$overallScore->{ $sys }{ 'documents' }{ $doc }{ 'segments' }{ $seg }{ 'score' } = $segScore;
			if ( $detail >= 2 )
			{
				printf "  $method score using 5-grams = %.4f for system \"$sys\" on segment $seg of document \"$doc\" (%d words)\n", $segScore, $tst_cnt->[1];
			}
		}
		$cum_ref_length += $ref_length;
		for (my $j=1; $j<=$max_Ngram; $j++)
		{
			$cum_match[$j] += $match_cnt->[$j];
			$cum_tst_cnt[$j] += $tst_cnt->[$j];
			$cum_ref_cnt[$j] += $ref_cnt->[$j];
			$cum_tst_info[$j] += $tst_info->[$j];
			$cum_ref_info[$j] += $ref_info->[$j];
		}
	}
	return ($cum_ref_length, [@cum_match], [@cum_tst_cnt], [@cum_ref_cnt], [@cum_tst_info], [@cum_ref_info]);
}

###############################################################################################################################
# function returning the shortest reference length
# takes as input:
#  - currentLength : the current (shortest) reference length
#  - referenceSentenceLength : the current reference sentence length
#  - candidateSentenceLength : the current candidate sentence length (unused)
###############################################################################################################################
sub brevity_penalty_shortest
{
	my ( $currentLength, $referenceSentenceLength, $candidateSentenceLength ) = @_;
	return ( $referenceSentenceLength < $currentLength ? $referenceSentenceLength : $currentLength );
}

###############################################################################################################################
# function returning the closest reference length (to the candidate sentence length)
# takes as input:
#  - currentLength: the current (closest) reference length.
#  - candidateSentenceLength : the current reference sentence length
#  - candidateSentenceLength : the current candidate sentence length
# when two reference sentences are at the same distance, it will return the shortest reference sentence length
# example of 4 iterations, given:
#  - one candidate sentence containing 7 tokens
#  - one reference translation containing 11 tokens
#  - one reference translation containing 8 tokens
#  - one reference translation containing 6 tokens
#  - one reference translation containing 7 tokens
# the multiple invokations will return:
#  - currentLength is set to 11 (outside of this function)
#  - brevity_penalty_closest( 11, 8, 7 ) returns 8, since abs( 8 - 7 ) < abs( 11 - 7 )
#  - brevity_penalty_closest( 8, 6, 7 ) returns 6, since abs( 8 - 7 ) == abs( 6 - 7 ) AND 6 < 8
#  - brevity_penalty_closest( 7, 6, 7 ) returns 7, since abs( 7 - 7 ) < abs( 6 - 7 )
###############################################################################################################################
sub brevity_penalty_closest
{
	my ( $currentLength, $referenceSentenceLength, $candidateSentenceLength ) = @_;
	my $result = $currentLength;
	if ( abs( $candidateSentenceLength - $referenceSentenceLength ) <= abs( $candidateSentenceLength - $currentLength ) )
	{
		if ( abs( $candidateSentenceLength - $referenceSentenceLength ) == abs( $candidateSentenceLength - $currentLength ) )
		{
			if ( $currentLength > $referenceSentenceLength )
			{
				$result = $referenceSentenceLength;
			}
		}
		else
		{
			$result = $referenceSentenceLength;
		}
	}
	return $result;
}

#################################

sub score_segment
{
	my ($tst_seg, @ref_segs) = @_;
	my (@tst_wrds, %tst_ngrams, @match_count, @tst_count, @tst_info);
	my (@ref_wrds, $ref_seg, %ref_ngrams, %ref_ngrams_max, @ref_count, @ref_info);
	my ($ngram);
	my (@nwrds_ref);
	my $ref_length;

	for (my $j=1; $j<= $max_Ngram; $j++)
	{
		$match_count[$j] = $tst_count[$j] = $ref_count[$j] = $tst_info[$j] = $ref_info[$j] = 0;
	}

# get the ngram counts for the test segment
	@tst_wrds = split /\s+/, $tst_seg;
	%tst_ngrams = %{Words2Ngrams (@tst_wrds)};
	for (my $j=1; $j<=$max_Ngram; $j++)
	{
		# compute ngram counts
		$tst_count[$j]  = $j<=@tst_wrds ? (@tst_wrds - $j + 1) : 0;
	}

# get the ngram counts for the reference segments
	foreach $ref_seg (@ref_segs)
	{
		@ref_wrds = split /\s+/, $ref_seg;
		%ref_ngrams = %{Words2Ngrams (@ref_wrds)};
		foreach $ngram (keys %ref_ngrams)
		{
			# find the maximum # of occurrences
			my @wrds = split / /, $ngram;
			$ref_info[@wrds] += $ngram_info{$ngram};
			$ref_ngrams_max{$ngram} = defined $ref_ngrams_max{$ngram} ? max ($ref_ngrams_max{$ngram}, $ref_ngrams{$ngram}) : $ref_ngrams{$ngram};
		}
		for (my $j=1; $j<=$max_Ngram; $j++)
		{
			# update ngram counts
			$ref_count[$j] += $j<=@ref_wrds ? (@ref_wrds - $j + 1) : 0;
		}
		if ( not defined( $ref_length ) )
		{
			$ref_length = scalar( @ref_wrds );
		}
		else
		{
			$ref_length = &{$BLEU_BP}( $ref_length, scalar( @ref_wrds ), scalar( @tst_wrds ) );
		}
	}

# accumulate scoring stats for tst_seg ngrams that match ref_seg ngrams
	foreach $ngram (keys %tst_ngrams)
	{
		next unless defined $ref_ngrams_max{$ngram};
		my @wrds = split / /, $ngram;
		$tst_info[@wrds] += $ngram_info{$ngram} * min($tst_ngrams{$ngram},$ref_ngrams_max{$ngram});
		$match_count[@wrds] += my $count = min($tst_ngrams{$ngram},$ref_ngrams_max{$ngram});
		printf "%.2f info for each of $count %d-grams = '%s'\n", $ngram_info{$ngram}, scalar @wrds, $ngram
			if $detail >= 3;
	}

	return ($ref_length, [@match_count], [@tst_count], [@ref_count], [@tst_info], [@ref_info]);
}

#################################

sub bleu_score_nosmoothing
{
	my ($ref_length, $matching_ngrams, $tst_ngrams, $sys, $SCOREmt) = @_;
	my $score = 0;
	my $iscore = 0;

	for ( my $j = 1; $j <= $max_Ngram; ++$j )
	{
		if ($matching_ngrams->[ $j ] == 0)
		{
			$SCOREmt->{ $j }{ $sys }{ cum }=0;
		}
		else
		{
			my $len_score = min (0, 1-$ref_length/$tst_ngrams->[1]);
			# Cumulative N-Gram score
			$score += log( $matching_ngrams->[ $j ] / $tst_ngrams->[ $j ] );
			$SCOREmt->{ $j }{ $sys }{ cum } = exp( $score / $j + $len_score );
			# Individual N-Gram score
			$iscore = log( $matching_ngrams->[ $j ] / $tst_ngrams->[ $j ] );
			$SCOREmt->{ $j }{ $sys }{ ind } = exp( $iscore );
		}
	}
	return $SCOREmt->{ 4 }{ $sys }{ cum };
}

###############################################################################################################################
# Default method used to compute the BLEU score, using smoothing.
# Note that the method used can be overridden using the '--no-smoothing' command-line argument
# The smoothing is computed by taking 1 / ( 2^k ), instead of 0, for each precision score whose matching n-gram count is null
# k is 1 for the first 'n' value for which the n-gram match count is null
# For example, if the text contains:
#   - one 2-gram match
#   - and (consequently) two 1-gram matches
# the n-gram count for each individual precision score would be:
#   - n=1  =>  prec_count = 2     (two unigrams)
#   - n=2  =>  prec_count = 1     (one bigram)
#   - n=3  =>  prec_count = 1/2   (no trigram,  taking 'smoothed' value of 1 / ( 2^k ), with k=1)
#   - n=4  =>  prec_count = 1/4   (no fourgram, taking 'smoothed' value of 1 / ( 2^k ), with k=2)
###############################################################################################################################
sub bleu_score
{
	my ($ref_length, $matching_ngrams, $tst_ngrams, $sys, $SCOREmt,$report_length) = @_;
	my $score = 0;
	my $iscore = 0;
	my $exp_len_score = 0;
	$exp_len_score = exp( min (0, 1 - $ref_length / $tst_ngrams->[ 1 ] ) ) if ( $tst_ngrams->[ 1 ] > 0 );
        print "length ratio: ".($tst_ngrams->[1]/$ref_length)." ($tst_ngrams->[1]/$ref_length), penalty (log): ".log($exp_len_score)."\n" if $report_length;
	my $smooth = 1;
	for ( my $j = 1; $j <= $max_Ngram; ++$j )
	{
		if ( $tst_ngrams->[ $j ] == 0 )
		{
			$iscore = 0;
		}
		elsif ( $matching_ngrams->[ $j ] == 0 )
		{
			$smooth *= 2;
			$iscore = log( 1 / ( $smooth * $tst_ngrams->[ $j ] ) );
		}
		else
		{
			$iscore = log( $matching_ngrams->[ $j ] / $tst_ngrams->[ $j ] );
		}
		$SCOREmt->{ $j }{ $sys }{ ind } = exp( $iscore );
		$score += $iscore;
		$SCOREmt->{ $j }{ $sys }{ cum } = exp( $score / $j ) * $exp_len_score;
	}
	return $SCOREmt->{ 4 }{ $sys }{ cum };
}

#################################

sub nist_score
{
	my ($nsys, $matching_ngrams, $tst_ngrams, $ref_ngrams, $tst_info, $ref_info, $sys, $SCOREmt) = @_;
	my $score = 0;
	my $iscore = 0;

	for (my $n=1; $n<=$max_Ngram; $n++)
	{
		$score += $tst_info->[$n]/max($tst_ngrams->[$n],1);
		$SCOREmt->{$n}{$sys}{cum} = $score * nist_length_penalty($tst_ngrams->[1]/($ref_ngrams->[1]/$nsys));
		$iscore = $tst_info->[$n]/max($tst_ngrams->[$n],1);
		$SCOREmt->{$n}{$sys}{ind} = $iscore * nist_length_penalty($tst_ngrams->[1]/($ref_ngrams->[1]/$nsys));
	}
	return $SCOREmt->{5}{$sys}{cum};
}

#################################

sub Words2Ngrams
{
	#convert a string of words to an Ngram count hash
	my %count = ();

	for (; @_; shift)
	{
		my ($j, $ngram, $word);
		for ($j=0; $j<$max_Ngram and defined($word=$_[$j]); $j++)
		{
			$ngram .= defined $ngram ? " $word" : $word;
			$count{$ngram}++;
		}
	}
	return {%count};
}

#################################

sub tokenization
{
	my ($norm_text) = @_;

# language-independent part:
	$norm_text =~ s/<skipped>//g; # strip "skipped" tags
	$norm_text =~ s/-\n//g; # strip end-of-line hyphenation and join lines
	$norm_text =~ s/\n/ /g; # join lines
	$norm_text =~ s/&quot;/"/g;  # convert SGML tag for quote to "
	$norm_text =~ s/&amp;/&/g;   # convert SGML tag for ampersand to &
	$norm_text =~ s/&lt;/</g;    # convert SGML tag for less-than to >
	$norm_text =~ s/&gt;/>/g;    # convert SGML tag for greater-than to <

# language-dependent part (assuming Western languages):
	$norm_text = " $norm_text ";
	$norm_text =~ tr/[A-Z]/[a-z]/ unless $preserve_case;
	$norm_text =~ s/([\{-\~\[-\` -\&\(-\+\:-\@\/])/ $1 /g;   # tokenize punctuation
	$norm_text =~ s/([^0-9])([\.,])/$1 $2 /g; # tokenize period and comma unless preceded by a digit
	$norm_text =~ s/([\.,])([^0-9])/ $1 $2/g; # tokenize period and comma unless followed by a digit
	$norm_text =~ s/([0-9])(-)/$1 $2 /g; # tokenize dash when preceded by a digit
	$norm_text =~ s/\s+/ /g; # one space only between words
	$norm_text =~ s/^\s+//;  # no leading space
	$norm_text =~ s/\s+$//;  # no trailing space

	return $norm_text;
}


sub tokenization_international
{
	my ($norm_text) = @_;

	$norm_text =~ s/<skipped>//g; # strip "skipped" tags
	$norm_text =~ s/\p{Hyphen}\p{Zl}//g; # strip end-of-line hyphenation and join lines
	$norm_text =~ s/\p{Zl}/ /g; # join lines

	# replace entities
	$norm_text =~ s/&quot;/\"/g;  # quote to "
	$norm_text =~ s/&amp;/&/g;   # ampersand to &
	$norm_text =~ s/&lt;/</g;    # less-than to <
	$norm_text =~ s/&gt;/>/g;    # greater-than to >
	$norm_text =~ s/&apos;/\'/g; # apostrophe to '

	$norm_text = lc( $norm_text ) unless $preserve_case; # lowercasing if needed
	$norm_text =~ s/([^[:ascii:]])/ $1 /g if ( $split_non_ASCII );

	# punctuation: tokenize any punctuation unless followed AND preceded by a digit
	$norm_text =~ s/(\P{N})(\p{P})/$1 $2 /g;
	$norm_text =~ s/(\p{P})(\P{N})/ $1 $2/g;

	$norm_text =~ s/(\p{S})/ $1 /g; # tokenize symbols

	$norm_text =~ s/\p{Z}+/ /g; # one space only between words
	$norm_text =~ s/^\p{Z}+//; # no leading space
	$norm_text =~ s/\p{Z}+$//; # no trailing space

	return $norm_text;
}

#################################

sub nist_length_penalty
{
	my ($ratio) = @_;
	return 1 if $ratio >= 1;
	return 0 if $ratio <= 0;
	my $ratio_x = 1.5;
	my $score_x = 0.5;
	my $beta = -log($score_x)/log($ratio_x)/log($ratio_x);
	return exp (-$beta*log($ratio)*log($ratio));
}

#################################

sub date_time_stamp
{
	my ($sec, $min, $hour, $mday, $mon, $year, $wday, $yday, $isdst) = localtime();
	my @months = qw(Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec);
	my ($date, $time);
	$time = sprintf "%2.2d:%2.2d:%2.2d", $hour, $min, $sec;
	$date = sprintf "%4.4s %3.3s %s", 1900+$year, $months[$mon], $mday;
	return ($date, $time);
}

#################################

sub extract_sgml_tag_and_span
{
	my ($name, $data) = @_;
	($data =~ m|<$name\s*([^>]*)>(.*?)</$name\s*>(.*)|si) ? ($1, $2, $3) : ();
}

#################################

sub extract_sgml_tag_attribute
{
	my ($name, $data) = @_;
	($data =~ m|$name\s*=\s*\"?([^\"]*)\"?|si) ? ($1) : ();
}

#################################

sub max
{
	my ($max, $next);

	return unless defined ($max=pop);
	while (defined ($next=pop))
	{
		$max = $next if $next > $max;
	}
	return $max;
}

#################################

sub min
{
	my ($min, $next);

	return unless defined ($min=pop);
	while (defined ($next=pop))
	{
		$min = $next if $next < $min;
	}
	return $min;
}

#################################

sub printout_report
{
	if ( $METHOD eq "BOTH" )
	{
		foreach my $sys (sort @tst_sys)
		{
			printf "NIST score = %2.4f  BLEU score = %.4f for system \"$sys\"\n",$NISTmt{5}{$sys}{cum},$BLEUmt{4}{$sys}{cum};
		}
	}
	elsif ($METHOD eq "NIST" )
	{
		foreach my $sys (sort @tst_sys)
		{
			printf "NIST score = %2.4f  for system \"$sys\"\n",$NISTmt{5}{$sys}{cum};
		}
	}
	elsif ($METHOD eq "BLEU" )
	{
		foreach my $sys (sort @tst_sys)
		{
			printf "\nBLEU score = %.4f for system \"$sys\"\n",$BLEUmt{4}{$sys}{cum};
		}
	}
	printf "\n# ------------------------------------------------------------------------\n\n";
	printf "Individual N-gram scoring\n";
	printf "        1-gram   2-gram   3-gram   4-gram   5-gram   6-gram   7-gram   8-gram   9-gram\n";
	printf "        ------   ------   ------   ------   ------   ------   ------   ------   ------\n";

	if ( ( $METHOD eq "BOTH" ) || ($METHOD eq "NIST") )
	{
		foreach my $sys (sort @tst_sys)
		{
			printf " NIST:";
			for (my $i=1; $i<=$max_Ngram; $i++)
			{
				printf "  %2.4f ",$NISTmt{$i}{$sys}{ind}
			}
			printf " \"$sys\"\n";
		}
		printf "\n";
	}

	if ( ( $METHOD eq "BOTH" ) || ($METHOD eq "BLEU") )
	{
		foreach my $sys (sort @tst_sys)
		{
			printf " BLEU:";
			for (my $i=1; $i<=$max_Ngram; $i++)
			{
				printf "  %2.4f ",$BLEUmt{$i}{$sys}{ind}
			}
			printf " \"$sys\"\n";
		}
	}

	printf "\n# ------------------------------------------------------------------------\n";
	printf "Cumulative N-gram scoring\n";
	printf "        1-gram   2-gram   3-gram   4-gram   5-gram   6-gram   7-gram   8-gram   9-gram\n";
	printf "        ------   ------   ------   ------   ------   ------   ------   ------   ------\n";

	if (( $METHOD eq "BOTH" ) || ($METHOD eq "NIST"))
	{
		foreach my $sys (sort @tst_sys)
		{
			printf " NIST:";
			for (my $i=1; $i<=$max_Ngram; $i++)
			{
				printf "  %2.4f ",$NISTmt{$i}{$sys}{cum}
			}
			printf " \"$sys\"\n";
		}
	}
	printf "\n";
	if ( ( $METHOD eq "BOTH" ) || ($METHOD eq "BLEU") )
	{
		foreach my $sys (sort @tst_sys)
		{
			printf " BLEU:";
			for (my $i=1; $i<=$max_Ngram; $i++)
			{
				printf "  %2.4f ",$BLEUmt{$i}{$sys}{cum}
			}
			printf " \"$sys\"\n";
		}
	}
}

###############################################################################################################################
# Create three files, by using:
# - $prefix : the prefix used for the output file names
# - %overall : a hash containing seg/doc/sys-level scores:
#   - $overall{ $SYSTEM_ID }{ 'score' } => system-level score
#   - $overall{ $SYSTEM_ID }{ 'documents' }{ $DOCUMENT_ID }{ 'score' } => document-level score
#   - $overall{ $SYSTEM_ID }{ 'documents' }{ $DOCUMENT_ID }{ 'segments' }{ $SEGMENT_ID } => segment-level score
###############################################################################################################################
sub outputMetricsMATR
{
	my ( $prefix, %overall ) = @_;
	my $fileNameSys = $prefix . '-sys.scr';
	my $fileNameDoc = $prefix . '-doc.scr';
	my $fileNameSeg = $prefix . '-seg.scr';
	open FILEOUT_SYS, '>', $fileNameSys or die "Could not open file: ${fileNameSys}";
	open FILEOUT_DOC, '>', $fileNameDoc or die "Could not open file: ${fileNameDoc}";
	open FILEOUT_SEG, '>', $fileNameSeg or die "Could not open file: ${fileNameSeg}";
	foreach my $sys ( sort( keys( %overall ) ) )
	{
		my $scoreSys = $overall{ $sys }{ 'score' };
		print FILEOUT_SYS "${tst_id}\t${sys}\t${scoreSys}\n";
		foreach my $doc ( sort( keys( %{$overall{ $sys }{ 'documents' }} ) ) )
		{
			my $scoreDoc = $overall{ $sys }{ 'documents' }{ $doc }{ 'score' };
			print FILEOUT_DOC "${tst_id}\t${sys}\t${doc}\t${scoreDoc}\n";
			foreach my $seg ( sort{ $a <=> $b }( keys( %{$overall{ $sys }{ 'documents' }{ $doc }{ 'segments' }} ) ) )
			{
				my $scoreSeg = $overall{ $sys }{ 'documents' }{ $doc }{ 'segments' }{ $seg }{ 'score' };
				print FILEOUT_SEG "${tst_id}\t${sys}\t${doc}\t${seg}\t${scoreSeg}\n";
			}
		}
	}
	close FILEOUT_SEG;
	close FILEOUT_DOC;
	close FILEOUT_SYS;
}

