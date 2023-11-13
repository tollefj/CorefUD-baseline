#!/bin/sh

# This file is based on CorPipe <https://github.com/ufal/crac2022-corpipe>.
# Copyright 2022 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University in Prague, Czech Republic.
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

curl --remote-name-all https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-5053{/CorefUD-1.1-public.zip}
unzip CorefUD-1.1-public.zip
for f in CorefUD-1.1-public/data/*/*.conllu; do
  lang=$(basename $f)
  lang=${lang%%-*}
  mkdir -p $lang
  mv $f $lang/$(basename $f)
done
rm -r CorefUD-1.1-public/ CorefUD-1.1-public.zip

mkdir dev
(cd dev
curl --remote-name-all https://ufal.mff.cuni.cz/~mnovak/files/corefud-1.1{/dev-blind.zip}
unzip dev-blind.zip
for f in *.conllu; do
  lang=${f%%-*}
  mv $f ../$lang/${f%%.*}.blind.conllu
done
)
rm -r dev/

mkdir test
  (cd test
  curl --remote-name-all https://ufal.mff.cuni.cz/~mnovak/files/corefud-1.1{/test-blind.zip}
  unzip test-blind.zip
  for f in *.conllu; do
    lang=${f%%-*}
    mv $f ../$lang/${f%%.*}.blind.conllu
  done
  )
rm -r test/
