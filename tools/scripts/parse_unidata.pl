#!/bin/perl

# script to generate unidata.lua from UnicodeData.txt

use strict;
use utf8;

my $cmode=(shift @ARGV) eq 'C';

if ($cmode) {
  print <<___HEADER;
#include "onmt/unicode/Data.h"

namespace onmt
{
  namespace unicode
  {
___HEADER
} else {
  print <<___HEADER;
  --[[ Class containing unicode data table summarized information]]
  local unidata = {}
  unidata.maplower = {}

  -- the following sections have been generated from UnicodeData.txt and should not be modified manually
___HEADER
}

my @Number;
my @Separator;
my @LetterLower;
my @LetterUpper;
my @LetterOther;
my @Mark;
my %maplower;

while(my $l=<STDIN>) {
  #0043;LATIN CAPITAL LETTER C;Lu;0;L;;;;;N;;;;0063;
  my ($codepoint,$name,$gc,$ccc,$bidicat,$cdm,$dgvalue,$dvalue,$nvalue,$mirrored,$u1name,$comment,$umap,$lmap,$tmap,$other1)=split(/;/,$l);
  die $l if (!defined $tmap or defined $other1);
  my $cp=eval "0x$codepoint";
  last if ($cp>65536);
  if ($gc=~/N/) { $Number[$cp]=1; }
  elsif ($gc=~/Z/) { $Separator[$cp]=1; }
  elsif ($gc=~/Ll/) { $LetterLower[$cp]=1; }
  elsif ($gc=~/Lu/) { $LetterUpper[$cp]=1; }
  elsif ($gc=~/L/) { $LetterOther[$cp]=1; }
  elsif ($gc=~/M/) { $Mark[$cp]=1; }
  if ($gc=~/L/ && $lmap ne "") {
    my $lower=eval "0x$lmap";
    $maplower{$cp}=$lower;
  }
}

if ($cmode) {
  generateC("Separator",@Separator);
  generateC("LetterLower",@LetterLower);
  generateC("Number",@Number);
  generateC("LetterOther",@LetterOther);
  generateC("LetterUpper",@LetterUpper);
  generateC("Mark",@Mark);

  print "\n    typedef std::unordered_map<code_point_t, code_point_t> map_unicode;\n";
  print "    const map_unicode map_lower={";
  my $first=1;
  my $count=0;
  foreach my $cp (sort {$a<=>$b} keys(%maplower)) {
    print "," if (!$first);
    if ($count%5==0) { print "\n                  "; }
    $count++;
    print "{".ghex($cp).",".ghex($maplower{$cp})."}";
    $first=0;
  }
  print "};\n";

  print <<___FOOTER;

  }
}
___FOOTER
}
else {
  generate("Number",@Number);
  generate("Separator",@Separator);
  generate("LetterLower",@LetterLower);
  generate("LetterUpper",@LetterUpper);
  generate("LetterOther",@LetterOther);
  generate("Mark",@Mark);

  foreach my $cp (sort {$a<=>$b} keys(%maplower)) {
    print "unidata.maplower[".ghex($cp)."]=".ghex($maplower{$cp})."\n";
  }

  print "\nreturn unidata\n";
}

sub generate {
  my ($s,@t)=@_;
  print "\nunidata.".$s." = {}\n";
  for(my $i=32; $i<=65536; $i++) {
    if ($t[$i]) {
      my $last=$i;
      my $j=$i+1;
      while($j-$last<128) {
        $last=$j if ($t[$j]);
        $j++;
      }
      print "unidata.".$s."[".ghex($i)."] = {";
      my $h=$i;
      my $C=0;
      my $firstline=1;
      while($h<=$last) {
        my $V=0;
        for(my $c=0;$c<16;$c++) {
          $V=$V<<1;
          $V=$V+1 if ($t[$h+$c]);
        }
        if ($C==0) { print "," if (!$firstline); print "\n  "; } else { print ", "; }
        print ghex($V);
        $C++;
        if ($C==10) { $C=0; $firstline=0; }
        $h=$h+16;
      }
      print "\n}\n";
      $i=$j;
    }
  }
}

sub generateC {
  my ($s,@t)=@_;
  print "\n    const map_of_list_t unidata_".$s." = {";
  my $first=1;
  for(my $i=32; $i<=65536; $i++) {
    if ($t[$i]) {
      my $last=$i;
      my $j=$i+1;
      while($j-$last<128) {
        $last=$j if ($t[$j]);
        $j++;
      }
      print "," if (!$first);
      $first=0;
      print "\n      {".ghex($i).", {";
      my $h=$i;
      my $C=0;
      my $firstline=1;
      while($h<=$last) {
        my $V=0;
        for(my $c=0;$c<16;$c++) {
          $V=$V<<1;
          $V=$V+1 if ($t[$h+$c]);
        }
        if ($C==0) { print "," if (!$firstline); print "\n          "; } else { print ", "; }
        print ghex($V);
        $C++;
        if ($C==10) { $C=0; $firstline=0; }
        $h=$h+16;
      }
      print "\n        }}";
      $i=$j-1;
    }
  }
  print "\n    };\n";
}

sub ghex {
  my ($v)=@_;
  return "0x".uc(sprintf("%04x",$v));
}
