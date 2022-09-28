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
    .setTitle('KC30-3fl-2uni (GPU RTX2060)')

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
    {x: 2174900, y: 2129648, z: 2185036},
    {x: 2136424, y: 2146750, z: 2163352},
    {x: 2150352, y: 2126584, z: 2148936},
    {x: 2111354, y: 2099142, z: 2106098},
    {x: 2119954, y: 2109506, z: 2181760},
    {x: 2181520, y: 2139654, z: 2164646},
    {x: 2174626, y: 2129548, z: 2160236},
    {x: 2167472, y: 2125360, z: 2151580},
    {x: 2135458, y: 2091788, z: 2179784},
    {x: 2172384, y: 2128092, z: 2137918},
    {x: 2206600, y: 2150022, z: 2189034},
    {x: 2217978, y: 2161734, z: 2141480},
    {x: 2144522, y: 2162654, z: 2156024},
    {x: 2205058, y: 2172396, z: 2230862},
    {x: 2189306, y: 2125156, z: 2160610},
    {x: 2161914, y: 2154546, z: 2142656},
    {x: 2128714, y: 2110626, z: 2199456},
    {x: 2136560, y: 2130294, z: 2150460},
    {x: 2150964, y: 2114048, z: 2128844},
    {x: 2136754, y: 2131338, z: 2175256},
    {x: 2153458, y: 2111850, z: 2141086},
    {x: 2183976, y: 2164238, z: 2181780},
    {x: 2193760, y: 2164808, z: 2200508},
    {x: 2145238, y: 2128882, z: 2098128},
    {x: 2187474, y: 2130502, z: 2156058},
    {x: 2174400, y: 2124946, z: 2161134},
    {x: 2162524, y: 2113940, z: 2207080},
    {x: 2192482, y: 2179800, z: 2186612},
    {x: 2156592, y: 2149502, z: 2147964},
    {x: 2149130, y: 2187890, z: 2131986},
    {x: 2120206, y: 2096210, z: 2118884},
    {x: 2210814, y: 2166058, z: 2178372},
    {x: 2087604, y: 2138556, z: 2135340},
    {x: 2160444, y: 2151350, z: 2142790},
    {x: 2191962, y: 2143774, z: 2159282},
    {x: 2176040, y: 2146144, z: 2150436},
    {x: 2111976, y: 2129260, z: 2146724},
    {x: 2111030, y: 2180214, z: 2142456},
    {x: 2165810, y: 2182542, z: 2174052},
    {x: 2165976, y: 2131410, z: 2166804},
    {x: 2188830, y: 2133168, z: 2185268},
    {x: 2183804, y: 2163008, z: 2190218},
    {x: 2165416, y: 2071064, z: 2191928},
    {x: 2179070, y: 2127270, z: 2139794},
    {x: 2152212, y: 2062888, z: 2129490},
    {x: 2153318, y: 2143612, z: 2187702},
    {x: 2197990, y: 2145406, z: 2139320},
    {x: 2216698, y: 2169798, z: 2225708},
    {x: 2109522, y: 2086660, z: 2136406},
    {x: 2159296, y: 2162764, z: 2220082},
    {x: 2125586, y: 2164582, z: 2123588},
    {x: 2183700, y: 2154342, z: 2212398},
    {x: 2179184, y: 2104476, z: 2161508},
    {x: 2182388, y: 2141000, z: 2196488},
    {x: 2141530, y: 2137820, z: 2123598},
    {x: 2114726, y: 2058832, z: 2128472},
    {x: 2124778, y: 2146102, z: 2120988},
    {x: 2189190, y: 2152808, z: 2177400},
    {x: 2102048, y: 2108586, z: 2162654},
    {x: 2143670, y: 2152500, z: 2190238},
    {x: 2186288, y: 2198976, z: 2204856},
    {x: 2185110, y: 2193944, z: 2204248},
    {x: 2103036, y: 2088310, z: 2129222},
    {x: 2174848, y: 2184166, z: 2142740}
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
    {x: 2095784, y: 2162978, z: 2020130},
    {x: 2097638, y: 2019176, z: 2076598},
    {x: 2018432, y: 2081486, z: 2098834},
    {x: 2038418, y: 2021950, z: 2103686},
    {x: 2058044, y: 1998766, z: 2090384},
    {x: 2050294, y: 2115976, z: 2055292},
    {x: 2062710, y: 2030688, z: 2057646},
    {x: 2027568, y: 2053954, z: 2081866},
    {x: 2055220, y: 2073714, z: 2051098},
    {x: 2058178, y: 2054746, z: 2026072},
    {x: 2094738, y: 2021910, z: 2089488},
    {x: 2037574, y: 2102662, z: 2061786},
    {x: 2047144, y: 2068826, z: 2067648},
    {x: 2053412, y: 2055596, z: 2062350},
    {x: 2032576, y: 2091922, z: 2070222},
    {x: 2044036, y: 2094578, z: 2060836},
    {x: 2032504, y: 2076398, z: 2074212},
    {x: 2048072, y: 2027506, z: 2088402},
    {x: 2047552, y: 2082902, z: 2061412},
    {x: 2048450, y: 2081984, z: 2066178},
    {x: 2047552, y: 2082902, z: 2061412},
    {x: 2094738, y: 2021910, z: 2089488},
    {x: 2048450, y: 2081984, z: 2066178},
    {x: 2051964, y: 2081682, z: 2070846},
    {x: 2077810, y: 2116328, z: 2028696},
    {x: 2074196, y: 2163328, z: 2040806},
    {x: 2148052, y: 2035392, z: 2095284},
    {x: 2027648, y: 2076208, z: 2101070},
    {x: 2097168, y: 2014524, z: 2154030},
    {x: 2046944, y: 2035908, z: 2107646},
    {x: 2078272, y: 2026470, z: 2146406},
    {x: 2129500, y: 2018904, z: 2106340}
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
    {x: 1841462, y: 1979288, z: 1989326},
    {x: 1851358, y: 1976198, z: 2027624},
    {x: 1858112, y: 1960856, z: 1979836},
    {x: 1867300, y: 1945350, z: 2033478},
    {x: 1873400, y: 1982872, z: 1953418},
    {x: 1880134, y: 1948234, z: 1945192},
    {x: 1885798, y: 1934796, z: 1964024},
    {x: 1885798, y: 1934796, z: 1964024},
    {x: 1887336, y: 1940870, z: 1941286},
    {x: 1887336, y: 1940870, z: 1941286},
    {x: 1891450, y: 1907450, z: 1963350},
    {x: 1894994, y: 1986844, z: 1936870},
    {x: 1900330, y: 1923406, z: 1960796},
    {x: 1909780, y: 1898460, z: 1953668},
    {x: 1915818, y: 1894614, z: 1982036},
    {x: 1918124, y: 1922522, z: 1924544},
    {x: 1919540, y: 1874122, z: 1977164},
    {x: 1933046, y: 1920100, z: 1949002},
    {x: 1934062, y: 1886374, z: 1970192},
    {x: 1941200, y: 1931816, z: 1914350},
    {x: 1941842, y: 1864912, z: 2019778},
    {x: 1949174, y: 1994236, z: 1904762},
    {x: 1950046, y: 1989554, z: 1890078},
    {x: 1950046, y: 1989554, z: 1890078},
    {x: 1953456, y: 2040948, z: 1861836},
    {x: 1958788, y: 1968894, z: 1895956},
    {x: 1972476, y: 1849420, z: 2044866},
    {x: 1981128, y: 2004306, z: 1889248},
    {x: 1989074, y: 2021202, z: 1885494},
    {x: 2004668, y: 2014438, z: 1886246},
    {x: 2009744, y: 1996062, z: 1880622},
    {x: 2022524, y: 1828474, z: 2065718}
])


// Add LegendBox to chart.
chart3D.addLegendBox().setTitle("")
    // Dispose example UI elements automatically if they take too much space. This is to avoid bad UI on mobile / etc. devices.
    .setAutoDispose({
        type: 'max-width',
        maxWidth: 0.30,
    })
    .add(chart3D)