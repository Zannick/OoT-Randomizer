from collections import OrderedDict, defaultdict
import csv
import hashlib
import json
import logging
import os, os.path
import random
import cProfile
import pstats
import string
import sys
import time

from World import World
from State import State
from Spoiler import Spoiler
from DungeonList import create_dungeons
from Fill import distribute_items_restrictive
from Item import Item
from ItemPool import generate_itempool, songlist
from Hints import buildGossipHints, get_hint_area
from Utils import default_output_path, is_bundled, subprocess_args, data_path
from version import __version__
from Settings import Settings
from SettingsList import setting_infos, logic_tricks
from Rules import set_rules
from Main import create_playthrough, dummy_window, main, cosmetic_patch, generate
try:
    rewardlist = World.rewardlist
except AttributeError:
    # 4.0
    from ItemPool import rewardlist

# odf required for outputting to .ods
try:
    import odf
    import odf.namespaces
    from odf.element import Element
    from odf.opendocument import OpenDocumentSpreadsheet as ODS
    from odf.style import *
    from odf.number import NumberStyle, Number, Text
    from odf.text import P
    from odf.table import Table, TableColumn, TableRow, TableCell, CalculationSettings, NamedExpressions
    CALCEXTNS = 'urn:org:documentfoundation:names:experimental:calc:xmlns:calcext:1.0'
    odf.namespaces.nsdict[CALCEXTNS] = 'calcext'
    def ConditionalFormatList(**args):
        return Element(qname=(CALCEXTNS, 'conditional-formats'), **args)
    def ConditionalFormat(**args):
        return Element(qname=(CALCEXTNS, 'conditional-format'), **args)
    def Condition(**args):
        return Element(qname=(CALCEXTNS, 'condition'), **args)
    def ColorScale(**args):
        return Element(qname=(CALCEXTNS, 'color-scale'), **args)
    def ColorScaleEntry(**args):
        return Element(qname=(CALCEXTNS, 'color-scale-entry'), **args)
    def CalcAttributes(kw):
        return {(CALCEXTNS, k): v for k,v in kw.items()}
    def TableCellP(p=None, **kwargs):
        c = TableCell(**kwargs)
        if p:
            c.addElement(P(text=p))
        if 'valuetype' in kwargs:
            c.setAttrNS(CALCEXTNS, 'value-type', kwargs['valuetype'])
        return c
    def column_name(ci):
        if ci < 26: return string.ascii_uppercase[ci]
        return string.ascii_uppercase[ci // 26 - 1] + string.ascii_uppercase[ci % 26]
except ImportError:
    odf = None

logging.basicConfig(level=logging.DEBUG, filename="LOG", filemode="w", format="%(asctime)-15s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s")

# another idea: stats for items findable in sphere n being required

# deal with lineprof annotations
try:
    profile
except NameError:
    profile = lambda f: f


def write_reqitem_log(spoiler, outfilebase=None):
    # we want to generate a table showing:
    # - for each location
    #   - the likelihood of that location being required
    #   - the likelihood of that location holding each item
    # - for each item
    #   - the likelihood of that item being required
    #   - the likelihood of that item being at each location
    # - for each realm/region
    #   - the likelihood of there being 0, 1, 2, 3... major items in that region
    #   - TODO number of locations
    #   - TODO expected number / mean/var of major items
    #   - the likelihood of each item being in that region
    # The data we need from each spoiler is simply:
    #   a list of tuples (Realm, Region, Location, Item, is_required, is_major)
    # 

    # TODO: Additional data:
    # per realm/region: number of eligible locations

    # For this purpose we need to count the two Colossus chests as Spirit Temple
    data = [(get_hint_area(loc), loc.parent_region.name, loc.name, loc.item.name,
             loc in spoiler.all_required_locations[0],
             loc.item.majoritem or loc.item.bossreward)
            for loc in spoiler.worlds[0].get_locations()
            if loc.item and not loc.locked]
    if outfilebase:
        outfile = '%s_reqs.json' % (outfilebase,)
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        with open(outfile, 'w') as f:
            f.write('[\n')
            f.write(',\n'.join(json.dumps(d,separators=(',',':')) for d in data))
            f.write('\n]')
    return spoiler, data

def make_one(base_settings, window=dummy_window(), out=False):
    start = time.process_time()

    logger = logging.getLogger('')

    # copy settings
    settings = Settings(base_settings)

    allowed_tricks = {}
    for trick in logic_tricks.values():
        settings.__dict__[trick['name']] = trick['name'] in settings.allowed_tricks

    settings.load_distribution()

    settings.compress_rom = 'None'
    settings.world_count = 1
    settings.player_num = 1
    settings.create_spoiler = False  # skip playthrough generation

    logger.info('OoT Randomizer Version %s  -  Seed: %s\n\n', __version__, settings.seed)
    settings.remove_disabled()
    logger.info('(Original) Settings string: %s\n', settings.settings_string)
    random.seed(settings.numeric_seed)
    settings.resolve_random_settings(cosmetic=False)
    logger.debug(settings.get_settings_display())

    spoiler = generate(settings, window)


    if settings.hints == 'none':
        State.update_required_items(spoiler)
        for world in worlds:
            world.update_useless_areas(spoiler)
            #buildGossipHints(spoiler, world)
    window.update_progress(55)
    spoiler.build_file_hash()

    settings_string_hash = hashlib.sha1(settings.settings_string.encode('utf-8')).hexdigest().upper()[:5]
    outfilebase = os.path.join('reqs', settings_string_hash, settings.seed)

    for world in spoiler.worlds:
        for setting in world.settings.__dict__:
            world.settings.__dict__[setting] = world.__dict__[setting]

    if out:
        return write_reqitem_log(spoiler, outfilebase)
    else:
        return write_reqitem_log(spoiler)

with open('tests/accessible.sav') as f:
    accessible = json.load(f)

with open('tests/multiworld.sav') as f:
    multiworld = json.load(f)

with open('tests/tokensanity.sav') as f:
    tokensanity = json.load(f)

with open('tests/entrance3.sav') as f:
    entrances = json.load(f)

with open('data/presets_default.json') as f:
    presets = json.load(f)
# presets[preset_name]
s3 = presets['S3 Tournament']

def merge_with(d, jsonfile):
    with open(jsonfile) as f:
        j = json.load(f)
    d['trials'] += j['trials']
    d['all_items'].update(j['all_items'])
    for item, n in j['item_reqs'].items():
        if item not in d['item_reqs']:
            d['item_reqs'][item] = n
        else:
            d['item_reqs'][item] += n
    for p in ('loc_items', 'reg_items', 'zone_items'):
        for area, area_dict in j[p].items():
            if area not in d[p]:
                d[p][area] = {}
            _ad = d[p][area]
            for k, n in area_dict.items():
                if k not in _ad:
                    _ad[k] = n
                else:
                    _ad[k] += n

def merge_all(dirname):
    jfs = [os.path.join(dirname, jf) for jf in os.listdir(dirname) if jf.endswith('.json') and jf != 'merged.json']
    obj = {'trials': 0, 'all_items': set(), 'item_reqs': {},
           'loc_items': {}, 'reg_items': {}, 'zone_items': {}}
    for jf in jfs:
        merge_with(obj, jf)
    obj['all_items'] = list(obj['all_items'])
    with open(os.path.join(dirname, 'merged.json'), 'w') as f:
        json.dump(obj, f)

def make_many(base_settings=s3, trials=1000):
    all_items = set()
    # item -> num times required
    item_reqs = defaultdict(int)
    # loc/reg/zone -> num required items -> num times
    # loc/reg/zone -> item -> num times
    loc_items = defaultdict(lambda: defaultdict(int))
    reg_items = defaultdict(lambda: defaultdict(int))
    zone_items = defaultdict(lambda: defaultdict(int))

    start = time.process_time()
    for t in range(trials):
        # todo: make this parallel!
        sp, data = make_one(base_settings)
        print('.', end='', flush=True)
        nreg = {}
        nzone = {}
        rreg = set()
        rzone = set()
        for zone, region, location, item, required, major in data:
            all_items.add(item)
            zone_items[zone][item] += 1
            reg_items[region][item] += 1
            loc_items[location][item] += 1
            if zone not in nzone: nzone[zone] = 0
            if region not in nreg: nreg[region] = 0
            if required:
                item_reqs[item] += 1
                loc_items[location]['required'] += 1
                # Regions are considered required iff a required major item is there.
                if major:
                    rreg.add(region)
                    rzone.add(zone)
            if major:
                loc_items[location]['1 major'] += 1
                nreg[region] += 1
                nzone[zone] += 1
        for reg, n in nreg.items():
            reg_items[reg]['%d major' % n if n < 3 else '3+ major'] += 1
        for zone, n in nzone.items():
            zone_items[zone]['%d major' % n if n < 3 else '3+ major'] += 1
        for reg in rreg:
            reg_items[reg]['required'] += 1
        for zone in rzone:
            zone_items[zone]['required'] += 1

    # todo: combine from previous file

    print('!', flush=True)
    print('Done in %s seconds (process time)' % (time.process_time() - start))
    settings = sp.worlds[0].settings 
    settings_string_hash = hashlib.sha1(settings.settings_string.encode('utf-8')).hexdigest().upper()[:5]
    outfilebase = os.path.join('reqs', settings_string_hash, settings.seed)
    outfilej = '%s_%d.json' % (outfilebase, trials)

    obj = {'trials': trials, 'all_items': list(all_items), 'item_reqs': item_reqs,
           'loc_items': loc_items, 'reg_items': reg_items, 'zone_items': zone_items}
    os.makedirs(os.path.dirname(outfilej), exist_ok=True)
    with open(outfilej, 'w') as f:
        json.dump(obj, f)
    print('Data saved to', outfilej)
    obj['all_items'] = all_items
    mfile = os.path.join('reqs', settings_string_hash, 'merged.json')
    if os.path.exists(mfile):
        merge_with(obj, mfile)
        outfilebase = os.path.join('reqs', settings_string_hash, 'merged')
        all_items = obj['all_items']
        obj['all_items'] = list(all_items)
        with open(mfile, 'w') as f:
            json.dump(obj, f)
        print('Merged data with', mfile)
        obj['all_items'] = all_items
    dump_csv(outfilebase, obj, settings_string_hash)

def dump_csv(outfilebase, obj, table_name):
    trials = obj['trials']
    all_items = obj['all_items']
    # turn dicts into defaultdict in case this is straight from a json object
    item_reqs = defaultdict(int, obj['item_reqs'])
    def dd(m):
        for x, v in m.items():
            m[x] = defaultdict(int, v)
        return m
    loc_items = dd(obj['loc_items'])
    reg_items = dd(obj['reg_items'])
    zone_items = dd(obj['zone_items'])
    outfile = '%s_%d.csv' % (outfilebase, trials)
    print('Tallying table data to', outfile)

    def _nreqs_key(d):
        return lambda area: (d[area]['required'], d[area]['3+ major'], d[area]['2 major'], d[area]['1 major'], d[area]['0 major'])

    def _ireqs_key(item):
        return (-item_reqs[item], item)

    field_names = ['Area', 'required', '0 major', '1 major', '2 major', '3+ major']
    # Add all the bossrewards, then all the songs, then the regular items,
    # each in descending order by overall required score (and then alphabetically)
    field_names.extend(sorted(rewardlist, key=_ireqs_key, reverse=True))
    field_names.extend(sorted(songlist, key=_ireqs_key, reverse=True))
    field_names.extend(sorted(
        all_items - set(field_names),
        key=_ireqs_key, reverse=True))

    with open(outfile, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=field_names)
        writer.writeheader()
        # Write item req scores as fractional values
        writer.writerow({i: n/trials for i,n in item_reqs.items()})
        # Write zones, regions, and locations
        for z in sorted(zone_items, key=_nreqs_key(zone_items), reverse=True):
            d = {'Area': z}
            d.update((i, n/trials) for i,n in zone_items[z].items())
            writer.writerow(d)
        writer.writerow({})
        for r in sorted(reg_items, key=_nreqs_key(reg_items), reverse=True):
            d = {'Area': r}
            d.update((i, n/trials) for i,n in reg_items[r].items())
            writer.writerow(d)
        writer.writerow({})
        for l in sorted(loc_items, key=lambda x:(loc_items[x]['required'], len(loc_items[x])), reverse=True):
            d = {'Area': l}
            d.update((i, n/trials) for i,n in loc_items[l].items())
            writer.writerow(d)

    if odf:
        csv_to_ods(outfile, table_name)

def extend_elements(targ, elist):
    for e in elist:
        targ.addElement(e)

def csv_to_ods(filename, tablename):
    with open(filename, newline='') as f:
        rows = list(csv.reader(f))

    items = rows[0][6:]
    # 3 additional rows for most-likely areas
    numrows = len(rows) + 3
    numcols = len(rows[0])

    # find the dividers, subtract 1 (and add 3) for the last row of the previous section
    lastrealm = lastregion = 0
    for i, row in enumerate(rows):
        if not any(row):
            if lastrealm:
                lastregion = i + 3
                break
            else:
                lastrealm = i + 3

    outfile = os.path.splitext(filename)[0] + '.ods'
    print('Building spreadsheet:', outfile)
    ods = ODS()
    # define styles
    fontname = "Droid Sans"
    font = FontFace(name=fontname, fontfamily=repr(fontname), fontfamilygeneric="swiss", fontpitch="variable")
    ods.fontfacedecls.addElement(font)
    base_style = DefaultStyle(family="table-cell")
    base_style.addElement(TextProperties(fontname=fontname, fontsize="11pt"))
    num = NumberStyle(name="N0")
    num.addElement(Number(minintegerdigits="1"))
    base2 = Style(name="Default", family="table-cell")
    zero = Style(name="zero", family="table-cell", parentstylename=base2)
    zero.addElement(TableCellProperties(backgroundcolor="#cc0000"))
    extend_elements(ods.styles, [base_style, num, base2, zero])

    cols = [150, 44, 42, 48, 80]
    for i, c in enumerate(cols):
        cx = Style(name="co%d" % (i+1), family="table-column")
        cx.addElement(TableColumnProperties(columnwidth="%dpt" % c, breakbefore="auto"))
        ods.automaticstyles.addElement(cx)

    row = Style(name="ro1", family="table-row")
    row.addElement(TableRowProperties(rowheight="12.81pt", breakbefore="auto", useoptimalrowheight="true"))
    row2 = Style(name="ro2", family="table-row")
    row2.addElement(TableRowProperties(rowheight="7.5pt", breakbefore="auto", useoptimalrowheight="false"))

    ce1 = Style(name="ce1", family="table-cell", parentstylename=base2)
    ce2 = Style(name="ce2", family="table-cell", parentstylename=base2)
    ce2.addElement(TableCellProperties(backgroundcolor="#000000"))
    ce3 = Style(name="ce3", family="table-cell", parentstylename=base2)
    ce3.addElement(Map(condition="cell-content()=0", applystylename="zero", basecelladdress="'%s'.B6" % tablename))
    extend_elements(ods.automaticstyles, [row, row2, ce1, ce2, ce3])

    # table here
    table = Table(name=tablename)
    cols = [1, 1, 3, 1, len(items)]
    for i, nc in enumerate(cols):
        table.addElement(TableColumn(stylename="co%d" % (i+1), defaultcellstylename=ce1, numbercolumnsrepeated=nc))

    header = TableRow(stylename="ro1")
    for h in rows[0]:
        header.addElement(TableCellP(valuetype="string", p=h))
    item_reqs = TableRow(stylename="ro1")
    item_reqs.addElement(TableCell(numbercolumnsrepeated=6))
    for r in rows[1][6:]:
        item_reqs.addElement(TableCellP(valuetype="float", value=r, stylename="ce3", p=r))

    ml_realm = TableRow(stylename="ro1")
    ml_realm.addElement(TableCellP(valuetype="string", p="Most likely realm"))
    ml_realm.addElement(TableCell(numbercolumnsrepeated=5))
    ml_region = TableRow(stylename="ro1")
    ml_region.addElement(TableCellP(valuetype="string", p="Most likely region"))
    ml_region.addElement(TableCell(numbercolumnsrepeated=5))
    ml_check = TableRow(stylename="ro1")
    ml_check.addElement(TableCellP(valuetype="string", p="Most likely check"))
    ml_check.addElement(TableCell(numbercolumnsrepeated=5))
    formula = "of:=INDEX([.$A${min}:.$A${max}]; MATCH(MAX([.{c}${min}:.{c}${max}]); [.{c}${min}:.{c}${max}]; 0); 1)"
    rec = "!RECALCULATE!"
    for ci in range(6, numcols):
        c = column_name(ci)
        ml_realm.addElement(TableCellP(
            formula=formula.format(min=6, max=lastrealm, c=c),
            valuetype="string", stringvalue=rec, p=rec))
        ml_region.addElement(TableCellP(
            formula=formula.format(min=lastrealm+2, max=lastregion, c=c),
            valuetype="string", stringvalue=rec, p=rec))
        ml_check.addElement(TableCellP(
            formula=formula.format(min=lastregion+2, max=numrows, c=c),
            valuetype="string", stringvalue=rec, p=rec))
    extend_elements(table, [header, item_reqs, ml_realm, ml_region, ml_check])

    for row in rows[2:]:
        blank = 0
        if not row[0]:
            r = TableRow(stylename="ro2")
            r.addElement(TableCell(stylename="ce2", numbercolumnsrepeated=numcols))
            table.addElement(r)
        else:
            r = TableRow(stylename="ro1")
            r.addElement(TableCellP(valuetype="string", p=row[0]))
            for ce in row[1:]:
                if ce:
                    if blank:
                        r.addElement(TableCell(stylename="ce3", numbercolumnsrepeated=blank))
                        blank = 0
                    r.addElement(TableCellP(stylename="ce3", valuetype="float", value=ce, p=ce))
                else:
                    blank += 1
            table.addElement(r)

    # conds here
    cfs = ConditionalFormatList()
    maxcol = column_name(numcols - 1)
    ranges = "'{0}'.G2:'{0}'.{1}2 '{0}'.B6:'{0}'.{1}{2} '{0}'.B{3}:'{0}'.{1}{4} '{0}'.B{5}:'{0}'.{1}{6}".format(
            tablename, maxcol, lastrealm, lastrealm + 2, lastregion, lastregion + 2, numrows)
    cf = ConditionalFormat(qattributes=CalcAttributes({"target-range-address": ranges}))
    cf.addElement(Condition(qattributes=CalcAttributes({
        "apply-style-name": "zero",
        "value": "=0",
        "base-cell-address": "'{0}'.B6".format(tablename)
    })))
    cfs.addElement(cf)
    cf2 = ConditionalFormat(qattributes=CalcAttributes({"target-range-address": ranges}))
    cs = ColorScale()
    cs.addElement(ColorScaleEntry(qattributes=CalcAttributes(
        {"value": ".00001", "type": "number", "color": "#e06666"})))
    cs.addElement(ColorScaleEntry(qattributes=CalcAttributes(
        {"value": ".25", "type": "number", "color": "#ffffff"})))
    cs.addElement(ColorScaleEntry(qattributes=CalcAttributes(
        {"value": "1", "type": "number", "color": "#38761d"})))
    cf2.addElement(cs)
    cfs.addElement(cf2)

    table.addElement(cfs, check_grammar=False)
    ods.spreadsheet.addElement(CalculationSettings(automaticfindlabels="false", useregularexpressions="false", usewildcards="true"))
    ods.spreadsheet.addElement(table)
    ods.spreadsheet.addElement(NamedExpressions())
    ods.save(outfile)

def profile_many(trials=1000, filename='ootprofile'):
    cProfile.run('make_many(trials={})'.format(trials), filename)
    return pstats.Stats(filename)


with open('tests/disables.sav') as f:
    dcount = json.load(f)

with open('dist.sav') as f:
    dist = json.load(f)

def m(settings):
    if settings.count != None and settings.count > 1:
        orig_seed = settings.seed
        for i in range(settings.count):
            settings.update_seed(orig_seed + '-' + str(i))
            main(settings)
    else:
        main(settings)

###
# remark out the below and run with python -i to use the REPL.
#
# copy the first output .json to merged.json in the same folder
# or manually run merge_all to get different runs combined together.
###

make_many(base_settings=s3,trials=250)
#main(Settings(multiworld))
#cProfile.run('m(Settings(multiworld))', 'oot3profile')
#cProfile.run('cosmetic_patch(Settings(cosmetics))', 'oot3profile')
#p = pstats.Stats('oot3profile')
#q = pstats.Stats('oot2profile')
#w = main(Settings(accessible))
#main(Settings(entrances))
