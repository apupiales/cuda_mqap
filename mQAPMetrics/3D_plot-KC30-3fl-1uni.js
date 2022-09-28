/*
 * LightningChartJS example that showcases PointSeries in a 3D Chart.
 *
 * https://lightningchart.com/lightningchart-js-interactive-examples/edit/lcjs-example-0900-3dScatter.html?theme=lightNew&page-theme=light
 * https://github.com/Arction/lcjs-example-0900-3dScatter/tree/master
 *
 */

// Import LightningChartJS
const lcjs = require('@arction/lcjs')

// Extract required parts from LightningChartJS.
const {
    lightningChart,
    SolidFill,
    ColorRGBA,
    PointStyle3D,
    Themes
} = lcjs

// Extract required parts from xyData.
const {
    createWaterDropDataGenerator
} = require('@arction/xydata')

// Initiate chart
const chart3D = lightningChart().Chart3D({
    disableAnimations: true,
    theme: Themes.lightNew,
})
    .setTitle('KC30-3fl-1uni (GPU RTX2060)')

// Set Axis titles
chart3D.getDefaultAxisX().setTitle('Objetivo 1')
chart3D.getDefaultAxisY().setTitle('Objetivo 2')
chart3D.getDefaultAxisZ().setTitle('Objetivo 3')

// Create Point Series for rendering max Y coords.
const basePopulation = chart3D.addPointSeries()
    .setPointStyle(new PointStyle3D.Triangulated({
        fillStyle: new SolidFill({ color: ColorRGBA(255, 215, 0) }),
        size: 10,
        shape: 'sphere'
    }))
    .setName('Población inicial')

basePopulation.add([
    {x: 2358508, y:2242690, z:2212456},
    {x: 2339842, y:2210576, z:2237644},
    {x: 2328482, y:2260398, z:2161982},
    {x: 2320732, y:2249630, z:2139108},
    {x: 2362368, y:2252918, z:2088270},
    {x: 2319518, y:2296178, z:2170098},
    {x: 2403916, y:2289350, z:2134716},
    {x: 2327226, y:2253512, z:2202884},
    {x: 2398196, y:2284742, z:2150606},
    {x: 2414492, y:2206248, z:2124076},
    {x: 2310812, y:2254446, z:2237512},
    {x: 2391594, y:2268746, z:2204342},
    {x: 2343568, y:2238882, z:2235188},
    {x: 2370352, y:2229308, z:2207728},
    {x: 2416934, y:2258118, z:2177754},
    {x: 2398936, y:2278820, z:2111832},
    {x: 2356832, y:2291672, z:2150244},
    {x: 2382444, y:2338162, z:2196346},
    {x: 2382568, y:2254864, z:2115034},
    {x: 2288144, y:2201342, z:2150846},
    {x: 2378922, y:2284336, z:2154406},
    {x: 2357804, y:2267818, z:2217566},
    {x: 2377458, y:2239172, z:2196146},
    {x: 2384082, y:2285574, z:2098904},
    {x: 2415932, y:2210332, z:2183072},
    {x: 2350634, y:2260772, z:2191766},
    {x: 2352970, y:2246468, z:2119954},
    {x: 2344662, y:2292430, z:2228996},
    {x: 2312732, y:2253866, z:2232350},
    {x: 2332336, y:2295036, z:2135140},
    {x: 2405264, y:2244310, z:2136972},
    {x: 2366562, y:2308252, z:2185414},
    {x: 2357064, y:2249250, z:2196478},
    {x: 2361582, y:2284674, z:2165234},
    {x: 2339112, y:2188052, z:2185036},
    {x: 2359148, y:2307468, z:2151074},
    {x: 2357628, y:2267828, z:2158096},
    {x: 2383492, y:2266478, z:2131512},
    {x: 2362400, y:2243950, z:2175344},
    {x: 2381322, y:2234538, z:2212320},
    {x: 2331310, y:2299262, z:2243540},
    {x: 2392850, y:2256448, z:2135064},
    {x: 2365514, y:2286010, z:2198234},
    {x: 2308862, y:2259962, z:2252808},
    {x: 2321630, y:2238696, z:2158382},
    {x: 2332416, y:2304550, z:2176884},
    {x: 2342274, y:2244736, z:2221084},
    {x: 2393304, y:2252022, z:2177688},
    {x: 2370644, y:2317946, z:2156262},
    {x: 2360996, y:2302678, z:2116560},
    {x: 2388430, y:2258292, z:2197716},
    {x: 2321118, y:2243924, z:2144812},
    {x: 2372430, y:2278594, z:2168156},
    {x: 2322146, y:2359094, z:2149458},
    {x: 2378930, y:2223732, z:2202388},
    {x: 2319958, y:2234334, z:2117946},
    {x: 2337718, y:2288982, z:2181538},
    {x: 2374134, y:2224368, z:2151842},
    {x: 2364060, y:2224700, z:2139454},
    {x: 2368922, y:2240902, z:2200302},
    {x: 2383190, y:2253490, z:2187448},
    {x: 2361074, y:2280362, z:2199318},
    {x: 2350226, y:2303386, z:2211340},
    {x: 2323544, y:2239730, z:2158230}
])


// Create another Point Series for rendering other Y coords than Max.
const nsga2 = chart3D.addPointSeries()
    .setPointStyle(new PointStyle3D.Triangulated({
        fillStyle: new SolidFill({ color: ColorRGBA(252, 116, 3) }),
        size: 10,
        shape: 'sphere'
    }))
    .setName('NSGA-II, 70 iteraciones')

nsga2.add([
    {x: 2228288, y: 2184498, z: 2141096},
    {x: 2236434, y: 2183142, z: 2151502},
    {x: 2240468, y: 2260114, z: 2112020},
    {x: 2246086, y: 2138984, z: 2154126},
    {x: 2255756, y: 2245870, z: 2121520},
    {x: 2265738, y: 2222826, z: 2130868},
    {x: 2268084, y: 2205650, z: 2107210},
    {x: 2276612, y: 2230092, z: 2091462},
    {x: 2283684, y: 2356430, z: 2071654},
    {x: 2284448, y: 2310510, z: 2066792},
    {x: 2286220, y: 2140934, z: 2092498},
    {x: 2287764, y: 2238156, z: 2082418},
    {x: 2293904, y: 2245998, z: 2029300},
    {x: 2296924, y: 2133020, z: 2102118},
    {x: 2302744, y: 2207660, z: 2082544},
    {x: 2308974, y: 2128646, z: 2117132},
    {x: 2312514, y: 2211646, z: 2077856},
    {x: 2313732, y: 2314566, z: 2009102},
    {x: 2317940, y: 2234122, z: 2049796},
    {x: 2322132, y: 2186130, z: 2091222},
    {x: 2330934, y: 2152470, z: 2085746},
    {x: 2346122, y: 2223774, z: 2055948},
    {x: 2352240, y: 2106696, z: 2135480},
    {x: 2352944, y: 2173448, z: 2078478},
    {x: 2359892, y: 2252364, z: 2003246},
    {x: 2368748, y: 2135834, z: 2087482},
    {x: 2371830, y: 2203358, z: 2057850},
    {x: 2375628, y: 2218448, z: 2054860},
    {x: 2385284, y: 2148620, z: 2082460},
    {x: 2389002, y: 2103848, z: 2132230},
    {x: 2404422, y: 2121434, z: 2129646},
    {x: 2410148, y: 2114532, z: 2127396}
])


// Create another Point Series for rendering other Y coords than Max.
const nsga2Greedy2opt = chart3D.addPointSeries()
    .setPointStyle(new PointStyle3D.Triangulated({
        fillStyle: new SolidFill({ color: ColorRGBA(192, 192, 192) }),
        size: 10,
        shape: 'sphere'
    }))
    .setName('NSGA-II+Greedy 2opt, 70 iteraciones')

nsga2Greedy2opt.add([
    {x: 2017758, y: 2229356, z: 2144590},
    {x: 2040284, y: 2218258, z: 2231516},
    {x: 2051846, y: 2181330, z: 2220404},
    {x: 2058658, y: 2321016, z: 2108504},
    {x: 2064734, y: 2258514, z: 2088082},
    {x: 2070944, y: 2195512, z: 2207754},
    {x: 2095068, y: 2304812, z: 2082038},
    {x: 2113598, y: 2110990, z: 2043696},
    {x: 2135486, y: 2093230, z: 2050268},
    {x: 2145718, y: 2088758, z: 2078470},
    {x: 2152190, y: 2117908, z: 2026108},
    {x: 2164674, y: 2052794, z: 2061920},
    {x: 2196110, y: 2150964, z: 1932686},
    {x: 2202738, y: 2036502, z: 2014958},
    {x: 2233324, y: 2030456, z: 1981704},
    {x: 2243232, y: 2097734, z: 1945838},
    {x: 2251508, y: 2007454, z: 2004196},
    {x: 2264744, y: 1980210, z: 2184002},
    {x: 2277936, y: 2075672, z: 1916844},
    {x: 2294458, y: 1998028, z: 2147172},
    {x: 2315116, y: 2228332, z: 1887940},
    {x: 2323354, y: 2251092, z: 1837234},
    {x: 2331962, y: 2247394, z: 1832788},
    {x: 2335002, y: 1971294, z: 2169330},
    {x: 2339324, y: 2232514, z: 1862212},
    {x: 2341822, y: 2173676, z: 1899962},
    {x: 2348206, y: 1959004, z: 2140212},
    {x: 2379942, y: 1948480, z: 2199184},
    {x: 2384344, y: 1936518, z: 2157038},
    {x: 2393264, y: 2208740, z: 1867612},
    {x: 2427534, y: 2174490, z: 1876864},
    {x: 2451446, y: 2207216, z: 1867428}
])


// Add LegendBox to chart.
chart3D.addLegendBox().setTitle("")
    // Dispose example UI elements automatically if they take too much space. This is to avoid bad UI on mobile / etc. devices.
    .setAutoDispose({
        type: 'max-width',
        maxWidth: 0.30,
    })
    .add(chart3D)