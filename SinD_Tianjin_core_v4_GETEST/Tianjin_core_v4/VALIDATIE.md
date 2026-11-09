# Verificatieverslag v4 (2026-11-09)

## Direct uitgevoerd

- 18 regressietests op de meegeleverde echte map-/lichtdata, synthetische grensgevallen en de acht eerder geëxporteerde replaytrajecten: geslaagd.
- Fasegroepen en replaytijdlijn uit de werkelijke `TrafficLight_8_2_1.csv`: gecontroleerd door programmatische vergelijking van alle acht tijdreeksen.
- Stoplijncoverage voor alle vier armen: virtuele detectielijn omvat de rijstrookcentra van de in-/uitgaande fysieke Lanelets nabij het kruispunt.
- Een volledige diagnostische run over **611 voertuigen en 129.273 trajectpunten** is uitgevoerd met het heropgebouwde **v3-exportbestand**, niet met het oorspronkelijke `Veh_smoothed_tracks.csv` en de echte metadata.
- Resultaat diagnostische run: 584 trajecten `classified`, 8 `classified_with_direction_caution`, 19 `partial_or_ambiguous`. Meer OD-toewijzingen dan v3, niet automatisch meer correcte OD-toewijzingen.
- Events in deze *diagnostische* run: 106 stabiele roodlichtkandidaten; 12 passages tijdens rood dicht bij faseovergang; 70 rechtsaf-bij-rood; 92 spookrijepisodes; 95 wachtperiodes tijdens rood.

## Vergelijking met bestaande SinD-labels (uit eerder v3-meta-exportrapport)

De 611 tracks zijn via `track_id` vergeleken met de eerder geëxporteerde SinD-metadata-annotaties (niet gebruikt tijdens detectie):

| SinD-label | Stabiel-rood-kandidaat | Geen stabiel-rood-kandidaat |
|---|---:|---:|
| `red-light running` (101) | 96 | 5 |
| `No violation of traffic lights` (495) | 10 | 485 |
| `yellow-light running` (15) | 0 | 15 |

**Alle vijf** niet als stabiel-rood gemarkeerde `red-light running` labels (track 17, 45, 285, 368, 549) hebben een roodfasepassage **binnen de onzekerheidsmarge rond de fasewisseling**. De tien extra stabiel-rood-kandidaten zonder overeenkomend label vereisen replaycontrole. De definities van modeluitvoer en metadata zijn niet gelijk; bereken uit deze tabel geen sensiviteit/specificiteit alsof de labels een onafhankelijke, complete gouden standaard zijn.

## Wat nog niet empirisch is aangetoond

- De volledige originele trajectbestanden plus originele lengte-/breedte-/oriëntatievelden zijn niet beschikbaar in deze werkomgeving (Git-LFS-pointers in de repositorykopie).
- Het werkelijke effect van de voertuigafmetingen op alle 611 trajecten is niet getest.
- De rijstrook- en spookrijdetectie is niet handmatig gelabeld voor de hele steekproef.
- De geografische OD-zones en virtuele stopstreeplijnen verdienen een steekproefsgewijze video-/kaartcontrole, vooral bij start-/eindtrajecten dicht bij het kruispunt.
- Geel, rechtsaf en fietslichttoepasselijkheid blijven volgens de afgesproken onderzoeksafbakening conservatief geïnterpreteerd.

Deze beperkingen zijn geen vooraf bekende programmeerfouten; het zijn ontbrekende onafhankelijke gegevens/validatie. De lokale run kan discrepancies en controleplots produceren die nodig zijn om ze te onderzoeken.
