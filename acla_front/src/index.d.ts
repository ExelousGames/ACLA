declare module "*.jpg";
declare module "*.jpeg";
declare module "*.png";
declare module "*.gif";
declare module "*.svg";

declare module "*.module.css" {
    const classes: { [key: string]: string };
    export default classes;
}

declare module "*.css";

declare namespace NodeJS {
    interface Require {
        context(
            directory: string,
            useSubdirectories: boolean,
            pattern: RegExp,
        ): {
            keys(): string[];
            (key: string): unknown;
        };
    }
}
