#!/usr/bin/perl 


use strict;
use utf8;
use open qw(:std :utf8);
#use English;
#$OUTPUT_AUTOFLUSH = 1;

#######################################################################
#Scoring Wrapper for QALB Shared Task
# This tool is a wrapper that requires the m2scorrer.py script to be 
# available. It normalizes Arabic text in terms of A/Y forms and 
# by removing punctuation marks.
#
# For any questions contact Nizar Habash (habash@ccls.columbia.edu)
# July 4, 2014
#######################################################################

my $argc  = @ARGV;
die "Usage: $0 <proposed_sentences> <gold_source_m2> (buckwalter|utf8)\n"
    if ($argc < 3);

my ($hyp,$gold,$encoding) = @ARGV;



&createNoPuncFiles($hyp,$gold,"$hyp.nopnx","$gold.nopnx");

&createNormFiles($hyp,$gold,"$hyp.norm","$gold.norm");

&createNormFiles("$hyp.nopnx","$gold.nopnx","$hyp.nopnx.norm","$gold.nopnx.norm");

print "========================\n";
print "Exact Match:\n";
print "python m2scorer.py $hyp $gold\n";
system("python2 m2scorer.py $hyp $gold");
print "========================\n";
print "Normalized A/Y:\n";
print "python m2scorer.py $hyp.norm $gold.norm\n";
system("python2 m2scorer.py $hyp.norm $gold.norm");
print "========================\n";
print "No Punctuation:\n";
print "python m2scorer.py $hyp.nopnx $gold.nopnx\n";
system("python2 m2scorer.py $hyp.nopnx $gold.nopnx");
print "========================\n";
print "No Punctuation and Normalized A/Y:\n";
print "python m2scorer.py $hyp.nopnx.norm $gold.nopnx.norm\n";
system("python2 m2scorer.py $hyp.nopnx.norm $gold.nopnx.norm");
print "========================\n";



#######################################################################
sub createNoPuncFiles { 
    my ($hyp,$gold,$hypn,$goldn)=@_;

    open(HYP, $hyp) || die "File $hyp not found\n";
    open(GLD, $gold)|| die "File $gold not found\n";
    open(HYPN, ">$hypn")|| die "File $hypn not opened\n";
    open(GLDN, ">$goldn")|| die "File $goldn not opened\n";

#Assume files are in BW

#For Hypothesis, we just remove all PNX


    while (my $hline = <HYP>){
	chomp($hline);
     	$hline=&noPunc($hline);
	print HYPN "$hline\n";
    }

#For Gold we need to:
    # a. remove all PNX in sentence
    # b. adjsut all indices per (a)
    # c. remove all adds, edit, deletes involving PNX

    my @gline=();
    my @glineorig=();
    my @gindex=();
    while (my $gline = <GLD>){
	chomp $gline;
	#print "OR: $gline\n";

	if ($gline=~/^S/){
	    $gline=~s/^S //;
	    @gline=split(' ',$gline);
	    @glineorig=split(' ',$gline);
	    for (my $g=0;$g<@gline; $g++){
		if (&isPunc($gline[$g])){
		    $gline[$g]="";
		    if ($g>0){
			$gindex[$g]=$gindex[$g-1];
		    }else{
			$gindex[$g]=-1;
		    }
		}else{
		     if ($g>0){
			$gindex[$g]=1+$gindex[$g-1];
		    }else{
			$gindex[$g]=0;
		    }
		}
	    }	
	    
	    #For final slot index:
	    $gindex[@gindex]=1+$gindex[@gindex-1];

	    $gline=join(' ',("S", @gline)); 
	    $gline=~s/\s+/ /g;
	    $gline=~s/\s+$//;
	    print GLDN "$gline\n";
	    
	    #print "$gline\n";
	    #for (my $g=0;$g<@gline; $g++){
		#print "$g => $gindex[$g] \t $gline[$g] / $glineorig[$g]\n";
	    #}

	}elsif($gline=~/^A /){
	    
	    my ($i,$j,$edit,$word,$fluff);
	    
	    if($gline=~/^A (\d+) (\d+)(\|\|\|[^\|]+\|\|\|)(.*)(\|\|\|REQUIRED.*)/){
		($i,$j,$edit,$word,$fluff)=($1,$2,$3,$4,$5);
	    }elsif($gline=~/^A (\d+) (\d+)(\|\|\|[^\|]+\|\|\|)(\|\|\|.*)/){
		($i,$j,$edit,$fluff)=($1,$2,$3,$4);
		$word="";
	    }

	    my $k=$j-1;
	    my $raw = join(' ',@glineorig[$i..$k]); 
	    #print "???? $raw\n";

	    if ((not &isPunc($raw))&&(not &isPunc($word))){
		$i=$gindex[$i];
		$j=$gindex[$j];
		print GLDN "A $i $j$edit$word$fluff\n";
		#print  ">>>A $i $j$edit$word$fluff\n";
	    }
	}else{
	    @gline=();
	    @glineorig=();
	    @gindex=();
	   
	    print GLDN "$gline\n";
	   
	}
	
    }

    close(HYP);
    close(GLD);
    close(HYPN);
    close(GLDN);
}

#######################################################################

sub isPunc {
    my ($str)=@_;
    if ((($encoding=~/buckwalter/i)&&($str=~/^[\.\,\;\?\!\"\:\(\)]+$/))
	||($str=~/^([.،؛؟\!\":\)\(]|[\.\,\;\?\!\"\:\(\)])+$/)){
	return(1);
    }else{
	return(0);
    }
}

sub noPunc {
    my ($str)=@_;
    if ($encoding=~/buckwalter/i){
	$str=~s/[\.\,\;\?\!\"\:\(\)]+//g;
    }else{
	$str=~s/([.،؛؟\!\":\)\(]|[\.\,\;\?\!\"\:\(\)])+//g;
    }
    $str=~s/\s+/ /g;
    return($str);
}

sub normalizeAY {
    my ($str)=@_;
    if ($encoding=~/buckwalter/i){
	$str=~s/[><\|]/A/g;
	$str=~s/Y/y/g;
    }else{
	$str=~s/[أإآ]/ا/g;
	$str=~s/ى/ي/g;
    }
    return($str);
}

#######################################################################

sub createNormFiles { 
    my ($hyp,$gold,$hypn,$goldn)=@_;

    open(HYP, $hyp) || die "File $hyp not found\n";
    open(GLD, $gold)|| die "File $gold not found\n";
    open(HYPN, ">$hypn")|| die "File $hypn not opened\n";
    open(GLDN, ">$goldn")|| die "File $goldn not opened\n";

#Assume files are in BW

    while (my $hline = <HYP>){
	$hline=&normalizeAY($hline);
	print HYPN $hline;
    }


    my @gline=();
    while (my $gline = <GLD>){
	#print $gline;
	
	chomp $gline;
	if ($gline=~/^S/){
	    $gline=&normalizeAY($gline);
	    print GLDN "$gline\n";
	    $gline=~s/^S //;

	    @gline=split(' ',$gline);
	}elsif($gline=~/^A /){
	    
	    $gline=~/^A (\d+) (\d+)(\|\|\|[^\|]+\|\|\|)(.*)(\|\|\|REQUIRED.*)/;
	    my ($i,$j,$edit,$word,$fluff)=($1,$2,$3,$4,$5);
	    
	    my $wnorm=&normalizeAY($word);
	    my $k=$j-1;
	    my $raw = join(' ',@gline[$i..$k]); 
	    
	    my $dist=$j-$i;
	    #print "$dist ($i,$j,$edit,$word,$fluff) $raw $wnorm\n";
	    
	    if ($raw ne $wnorm){
		print GLDN "A $i $j$edit$wnorm$fluff\n";
	    }
	}else{
	    @gline=();
	    print GLDN "$gline\n";
	}
	
    }

    close(HYP);
    close(GLD);
    close(HYPN);
    close(GLDN);
}

#######################################################################
