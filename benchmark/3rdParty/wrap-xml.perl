#!/usr/bin/env perl
#
# This file is part of moses.  Its use is licensed under the GNU Lesser General
# Public License version 2.1 or, at your option, any later version.

use warnings;
use strict;

my ($language,$src,$system) = @ARGV;
die("wrapping frame not found ($src)") unless -e $src;
$system = "Edinburgh" unless $system;

open(SRC,$src) or die "Cannot open: $!";
my @OUT = <STDIN>;
chomp(@OUT);
#my @OUT = `cat $decoder_output`;
my $missing_end_seg = 0;
while(<SRC>) {
    chomp;
    if (/^<srcset/) {
	s/<srcset/<tstset trglang="$language"/i;
    }
    elsif (/^<\/srcset/) {
	s/<\/srcset/<\/tstset/i;
    }
    elsif (/^<doc/i) {
  s/ *sysid="[^\"]+"//;
	s/<doc/<doc sysid="$system"/i;
    }
    elsif (/<seg/) {
	my $line = shift(@OUT);
        $line = "" if $line =~ /NO BEST TRANSLATION/;
        if (/<\/seg>/) {
	  s/(<seg[^>]+> *).*(<\/seg>)/$1$line$2/i;
          $missing_end_seg = 0;
        }
        else {
	  s/(<seg[^>]+> *)[^<]*/$1$line<\/seg>/i;
          $missing_end_seg = 1;
        }
    }
    elsif ($missing_end_seg) {
      if (/<\/doc>/) {
        $missing_end_seg = 0;
      }
      else {
        next;
      }
    }
    print $_."\n";
}
