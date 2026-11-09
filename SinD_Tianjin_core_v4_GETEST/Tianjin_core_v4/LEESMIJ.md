# SinD Tianjin 8_2_1 — Kernclassificatie v4

**Scope:** herkomst–bestemming (OD), roodlichtpassages, wachten t.o.v. de stopstreep en spookrijden. De oude detectoren voor afslagpijlen, rijstrookcategorieën, scherpe bochten en zebrapaden zijn **niet** actief. Het basisbestand `geometry_legacy_v3.py` dient uitsluitend voor het inlezen van de map/CSV's, bewegingsschatting, signaalparser en enkele geometrische hulpfuncties. Zijn oude `main()` en gebeurtenisdetectoren worden niet gebruikt.

## Bestanden plaatsen

Pak het pakket uit zodat dit bestaat:

```
SinD/
  Data/Tianjin/8_2_1/Veh_smoothed_tracks.csv
  Data/Tianjin/8_2_1/Veh_tracks_meta.csv
  Tianjin_core_v4/
    analyse_tianjin_core_v4.py
    geometry_legacy_v3.py
    TrafficLight_8_2_1.csv
    tianjin_lanelets_uitgebreid.osm
    test_core_v4.py
    replay_controlepunten_UIT_V3_EXPORT.csv
```

Open **SinD** (hoofdmap) in VS Code en voer deze commando's uit in de terminal:

```
python -m pip install numpy pandas shapely matplotlib
python -m unittest discover -s Tianjin_core_v4 -p test_core_v4.py -v
python Tianjin_core_v4/analyse_tianjin_core_v4.py --plots --plot-ids 1,2,4,9,10,11,12,17
```

De echte `Veh_smoothed_tracks.csv` en `Veh_tracks_meta.csv` moeten volledig gedownload zijn; Git-LFS-pointerbestanden zijn niet voldoende. De werkelijke signaaldata zitten in de ZIP. Het script zoekt data relatief vanuit de **huidige SinD-hoofdmap**.

## Resultaten

Standaardmap: `analysis_outputs/lanelet_audit/tianjin_core_v4/`.

- `per_voertuig.csv`: één rij per track, OD, signaal-/spookrijstatus, aantal wachtperiodes, annotatie.
- `gebeurtenissen.csv`: individuele passages, stilstand en spookrijepisodes, inclusief tijd, plaats en bewijssoort.
- `stoplijnen_en_virtuele_lijnen.csv`: fysieke stopstreepmodellen en virtuele verlengingen over de hele weg.
- `stoplijn_coverage.csv`: kaartdekking per in-/uitgaande Lanelet.
- `vergelijking_met_SinD_labels.csv`: kruistabel. **Geen bewezen confusion matrix.**
- `prioritaire_discrepanties.csv`: voertuigen waarvan roodlichtdetectie afwijkt van SinD-annotaties.
- `validatierapport.csv` en `.json`: uitgevoerde controles en onzekere aannames.
- `config_actief.json`: alle gekozen detectiedrempels.
- `plots/track_<ID>.png`: schematische trajecten voor geselecteerde IDs.

## Betekenis van labels

`red_light_running_candidate`: niet-rechtsafgaande beweging, **tijdens stabiel rood** de stopstreep/virtuele toegangslijn overgestoken. Dit is een *vermoedelijke* overtreding, geen juridische uitspraak.

`red_crossing_near_phase_transition`: passage tijdens rood binnen de gekozen marge van 350 ms rond een fasewisseling: **handmatige controle**. Niet stilzwijgend als groen of als zeker rood tellen.

`right_turn_on_red`: roodpassage, rechtsaf; volgens de afgesproken onderzoeksafbakening **niet** meetellen als roodlichtovertreding. Voorrangverlening wordt niet beoordeeld.

`wait_during_red`: geregistreerde stilstand vóór, op of voorbij stopstreep. **Nooit automatisch** een zelfstandige roodlichtovertreding. De passagegeschiedenis wordt toegevoegd als die werd waargenomen.

`wrong_way_candidate`: gedurende minimaal 0,3 s / 2 m / 3 meetpunten tegen de richting van een fysieke rijstrook bewogen; een loodrechte kruising van een strook telt niet mee.

`not_assessable_crossing_not_observed`: geen stopstreeppassage in de waargenomen trajectpunten; geen bewijs van reglementair gedrag.

## Onderzoeksafspraken en beperkingen

- N/S-fasegroep: verkeerslichten 1,4,5,8; E/W: 2,3,6,7. De echte tijdlijn 0–44 s wordt automatisch gecontroleerd. De individuele lamp op elk armvlak is niet afzonderlijk bewezen maar de groepsfase is identiek.
- Fietsers en andere niet-gemotoriseerde voertuigen gebruiken **volgens gebruikersaanname** dezelfde algemene lichten.
- Het uiteinde van de inkomende Lanelet, inclusief `approach_gap`, wordt **volgens gebruikersaanname** als stopstreep gebruikt.
- De fysieke stopstreep wordt apart bijgehouden van de virtuele verlenging naar de verkeerde weghelft. Enkel naderende passages worden geteld.
- Afmetingen uit originele metadata ondersteunen een benadering van de voorkant. De exacte positie van het annotatie-referentiepunt/oriëntatie van elk object is niet onafhankelijk bewezen.
- De fout uit v3 met een eis van >=0,25 m per tijdstap is verwijderd. In plaats daarvan zijn kleinste stabiele vóór/na-posities rond de lijn vereist (standaard 0,05 m binnen 2 s). Drempels zijn hypotheses en hebben een gevoeligheidsanalyse nodig.
- Dit is **geen gedocumenteerd rechtshandhavingssysteem**. Een traject dat rood voorbij een virtuele lijn passeert, vereist altijd interpretatie van rijstrook, afslag en context. Ook wat SinD als label geeft is niet automatisch volledig correcte grondwaarheid.

## Opties

```
python Tianjin_core_v4/analyse_tianjin_core_v4.py --help
```

Geef andere paden via `--tracks`, `--meta`, `--signals`, `--osm`, `--output-dir`. Alle detectorparameters kunnen via `--config custom.json` worden aangepast; keys staan in `DEFAULTS` bovenaan het script en worden in `config_actief.json` geëxporteerd.

### Alleen voor technische diagnose

Het meegeleverde `replay_controlepunten_UIT_V3_EXPORT.csv` bevat echte, eerder geëxporteerde punten van acht voertuigtrajecten; **geen volledige originele tracks/metadata**. Om daarop een oefenrun te doen:

```
python Tianjin_core_v4/analyse_tianjin_core_v4.py --tracks Tianjin_core_v4/replay_controlepunten_UIT_V3_EXPORT.csv --allow-missing-meta-for-diagnostics --output-dir analysis_outputs/diagnostic_v4
```

Die uitvoer mag **niet** als onafhankelijke validatie van lengte/oriëntatie of nauwkeurigheid worden beschouwd. De standaardrun vereist metadata; als dat bestand ontbreekt stopt de code.
