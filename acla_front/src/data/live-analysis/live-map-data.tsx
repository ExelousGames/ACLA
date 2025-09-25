//match acc shared memory to a name saved in database
export const ACCMemoeryTracks = new Map([
    ["brands_hatch", "Brands Hatch Circuit"],
    ["monza", "Autodromo Nazionale Monza"],
    ["spa", "Circuit de Spa-Francorchamps"],
    ["nurburgring", "Nürburgring"],
    ["silverstone", "Silverstone Circuit"],
    ["paul_ricard", "Circuit Paul Ricard"],
    ["misano", "Misano World Circuit"],
    ["barcelona", "Circuit de Barcelona-Catalunya"],
    ["hungaroring", "Hungaroring"],
    ["zandvoort", "Circuit Park Zandvoort"],
    ["zolder", "Circuit Zolder"],
    ["imola", "Autodromo Enzo e Dino Ferrari"],
    ["oulton_park", "Oulton Park"],
    ["donington", "Donington Park"],
    ["snetterton", "Snetterton Circuit"],
    ["watkins_glen", "Watkins Glen International"],
    ["cota", "Circuit of The Americas"],
    ["indianapolis", "Indianapolis Motor Speedway"],
    ["laguna_seca", "WeatherTech Raceway Laguna Seca"],
    ["kyalami", "Kyalami Grand Prix Circuit"],
    ["suzuka", "Suzuka Circuit"],
    ["mount_panorama", "Mount Panorama Circuit"],
    ["valencia", "Circuit Ricardo Tormo"],
]);


export enum ACC_STATUS {
    ACC_OFF, // 0
    ACC_REPLAY,  // 1
    ACC_LIVE, // 2
    ACC_PAUSE   // 3
}