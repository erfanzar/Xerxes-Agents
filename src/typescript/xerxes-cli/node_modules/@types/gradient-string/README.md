# Installation
> `npm install --save @types/gradient-string`

# Summary
This package contains type definitions for gradient-string (https://github.com/bokub/gradient-string).

# Details
Files were exported from https://github.com/DefinitelyTyped/DefinitelyTyped/tree/master/types/gradient-string.
## [index.d.ts](https://github.com/DefinitelyTyped/DefinitelyTyped/tree/master/types/gradient-string/index.d.ts)
````ts
import tinycolor = require("tinycolor2");

declare namespace gradient {
    interface PositionedColorInput {
        color: tinycolor.ColorInput;
        pos: number;
    }

    interface Gradient {
        (message?: string, opt?: Options): string;
        multiline(message?: string, opt?: Options): string;
    }

    interface Options {
        interpolation?: string | undefined;
        hsvSpin?: string | undefined;
    }

    const atlas: Gradient;
    const cristal: Gradient;
    const teen: Gradient;
    const mind: Gradient;
    const morning: Gradient;
    const vice: Gradient;
    const passion: Gradient;
    const fruit: Gradient;
    const instagram: Gradient;
    const retro: Gradient;
    const summer: Gradient;
    const rainbow: Gradient;
    const pastel: Gradient;
}

declare function gradient(colors: tinycolor.ColorInput[] | gradient.PositionedColorInput[]): gradient.Gradient;
declare function gradient(...colors: tinycolor.ColorInput[]): gradient.Gradient;
declare function gradient(...colors: gradient.PositionedColorInput[]): gradient.Gradient;
export = gradient;

````

### Additional Details
 * Last updated: Wed, 03 Apr 2024 18:35:48 GMT
 * Dependencies: [@types/tinycolor2](https://npmjs.com/package/@types/tinycolor2)

# Credits
These definitions were written by .
