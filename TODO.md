Just some notes...

 * for 1.0.0: more complete docstring coverage
 * for 1.0.0: fix fluid tracing to properly interpolate in time
 * for 1.0.0: tidy up the parallel module

Better uninstall... for now, use this:

where="~/Library/anaconda"
rm "$(ls ${where}/bin/{athena2xdmf,bitmaskbits,fnslice,rgb2hex,viscid*})"
rm -rf "$(ls ${where}/lib/python*/site-packages/viscid*)"
where="~/Library/anaconda/envs/*"
rm "$(ls ${where}/bin/{athena2xdmf,bitmaskbits,fnslice,rgb2hex,viscid*})"
rm -rf "$(ls ${where}/lib/python*/site-packages/viscid*)"
