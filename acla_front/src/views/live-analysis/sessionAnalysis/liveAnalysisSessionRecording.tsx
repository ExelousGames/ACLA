import { Card, Flex, Box, TextField, IconButton, Heading, Grid, Text, Slider, Avatar } from '@radix-ui/themes';
import { PythonShell } from 'python-shell';

import { Link } from 'react-router-dom';

const LiveAnalysisSessionRecording = () => {
    let tabIndex = 0
    return (
        <Card size="4" style={{ height: 896 }}>

            <Box
                position="absolute"
                left="0"
                right="0"
                bottom="0"
                mb="5"
                height="64px"
                style={{
                    borderRadius: "100px",
                    boxShadow: "var(--shadow-6)",
                    marginLeft: 200,
                    marginRight: 200,
                }}
            >
                <Box
                    width="100%"
                    height="100%"
                    position="absolute"
                    overflow="hidden"
                    style={{
                        borderRadius: "100px",
                        backgroundColor: "var(--color-floating-panel)",
                        filter: "saturate(0.5) brightness(1.1)",
                        WebkitBackdropFilter: "blur(24px)",
                        backdropFilter: "blur(24px)",
                    }}
                />
                <Flex height="100%" justify="between" position="relative">
                    <Flex gap="4" align="center" p="3">
                        <IconButton tabIndex={tabIndex} radius="full" size="3">
                            <svg
                                xmlns="http://www.w3.org/2000/svg"
                                fill="currentcolor"
                                viewBox="0 0 30 30"
                                width="20"
                                height="20"
                                style={{ marginRight: -2 }}
                            >
                                <path d="M 6 3 A 1 1 0 0 0 5 4 A 1 1 0 0 0 5 4.0039062 L 5 15 L 5 25.996094 A 1 1 0 0 0 5 26 A 1 1 0 0 0 6 27 A 1 1 0 0 0 6.5800781 26.8125 L 6.5820312 26.814453 L 26.416016 15.908203 A 1 1 0 0 0 27 15 A 1 1 0 0 0 26.388672 14.078125 L 6.5820312 3.1855469 L 6.5800781 3.1855469 A 1 1 0 0 0 6 3 z" />
                            </svg>
                        </IconButton>

                        <Flex align="center" gap="4">
                            <IconButton
                                tabIndex={tabIndex}
                                color="gray"
                                variant="ghost"
                                radius="full"
                                size="2"
                            >
                                <svg
                                    xmlns="http://www.w3.org/2000/svg"
                                    viewBox="0 0 30 30"
                                    width="20"
                                    height="20"
                                    fill="currentcolor"
                                    fillOpacity={0.7}
                                >
                                    <path d="M 21 5 L 21 8 L 18.675781 8 C 16.670448 8 14.796256 9.00408 13.683594 10.671875 L 12 13.197266 L 10.316406 10.671875 C 9.2045791 9.0047337 7.329552 8 5.3242188 8 L 3 8 A 1.0001 1.0001 0 1 0 3 10 L 5.3242188 10 C 6.6628853 10 7.910171 10.668391 8.6523438 11.78125 L 10.798828 15 L 8.6523438 18.21875 C 7.910171 19.331609 6.6628854 20 5.3242188 20 L 3 20 A 1.0001 1.0001 0 1 0 3 22 L 5.3242188 22 C 7.3295521 22 9.2045792 20.995266 10.316406 19.328125 L 12 16.802734 L 13.683594 19.328125 C 14.796256 20.99592 16.670448 22 18.675781 22 L 21 22 L 21 25 L 27 21 L 21 17 L 21 20 L 18.675781 20 C 17.337115 20 16.090994 19.332955 15.347656 18.21875 L 13.201172 15 L 15.347656 11.78125 C 16.090994 10.667045 17.337115 10 18.675781 10 L 21 10 L 21 13 L 27 9 L 21 5 z" />
                                </svg>
                            </IconButton>
                            <IconButton
                                tabIndex={tabIndex}
                                color="gray"
                                variant="ghost"
                                radius="full"
                                size="2"
                            >
                                <svg
                                    xmlns="http://www.w3.org/2000/svg"
                                    viewBox="0 0 30 30"
                                    fill="currentcolor"
                                    fillOpacity={0.7}
                                    width="20"
                                    height="20"
                                >
                                    <path d="M 20 4 L 20 7 L 8 7 C 4.6983746 7 2 9.6983746 2 13 A 1.0001 1.0001 0 1 0 4 13 C 4 10.779625 5.7796254 9 8 9 L 20 9 L 20 12 L 27 8 L 20 4 z M 26.984375 15.986328 A 1.0001 1.0001 0 0 0 26 17 C 26 19.220375 24.220375 21 22 21 L 10 21 L 10 18 L 3 22 L 10 26 L 10 23 L 22 23 C 25.301625 23 28 20.301625 28 17 A 1.0001 1.0001 0 0 0 26.984375 15.986328 z" />
                                </svg>
                            </IconButton>
                        </Flex>
                    </Flex>

                    <Flex align="center" gap="3">
                        <IconButton
                            tabIndex={tabIndex}
                            color="gray"
                            variant="ghost"
                            radius="full"
                            size="2"
                        >
                            <svg
                                xmlns="http://www.w3.org/2000/svg"
                                fill="currentcolor"
                                fillOpacity={0.7}
                                viewBox="0 0 30 30"
                                width="20"
                                height="20"
                                style={{ marginRight: 2 }}
                            >
                                <path d="M 14 6 A 1 1 0 0 0 13.408203 6.1953125 A 1 1 0 0 0 13.378906 6.2167969 L 2.4609375 14.15625 A 1 1 0 0 1 2.4589844 14.158203 L 2.4433594 14.169922 A 1 1 0 0 0 2 15 A 1 1 0 0 0 2.4492188 15.833984 L 13.40625 23.804688 A 1 1 0 0 0 14 24 A 1 1 0 0 0 15 23 A 1 1 0 0 0 15 22.996094 L 15 16.234375 L 25.40625 23.804688 A 1 1 0 0 0 26 24 A 1 1 0 0 0 27 23 A 1 1 0 0 0 27 22.996094 L 27 15 L 27 7.0039062 A 1 1 0 0 0 27 7 A 1 1 0 0 0 26 6 A 1 1 0 0 0 25.40625 6.1953125 A 1 1 0 0 0 25.378906 6.2167969 L 15 13.763672 L 15 7.0039062 A 1 1 0 0 0 15 7 A 1 1 0 0 0 14 6 z" />
                            </svg>
                        </IconButton>

                        <Flex align="center" gap="3">
                            <img
                                src={songsHistory[6].cover}
                                width="48"
                                height="48"
                                style={{ borderRadius: "var(--radius-2)" }}
                            />
                            <Box>
                                <Text size="1" as="div" weight="medium">
                                    {songsHistory[6].name}
                                </Text>
                                <Text size="1" as="div" color="gray" mb="2">
                                    {songsHistory[6].artist} – {songsHistory[6].album}
                                </Text>

                                <Box
                                    position="relative"
                                    height="4px"
                                    width="320px"
                                    style={{
                                        backgroundColor: "var(--gray-a5)",
                                        borderRadius: "var(--radius-1)",
                                    }}
                                >
                                    <Box
                                        position="absolute"
                                        height="4px"
                                        width="64px"
                                        style={{
                                            borderRadius: "var(--radius-1)",
                                            backgroundColor: "var(--gray-a9)",
                                        }}
                                    />
                                    <Box position="absolute" top="0" right="0" mt="-28px">
                                        <Text size="1" color="gray">
                                            0:58 / {songsHistory[6].length}
                                        </Text>
                                    </Box>
                                </Box>
                            </Box>
                        </Flex>

                        <IconButton
                            tabIndex={tabIndex}
                            color="gray"
                            variant="ghost"
                            radius="full"
                            size="2"
                        >
                            <svg
                                xmlns="http://www.w3.org/2000/svg"
                                viewBox="0 0 30 30"
                                width="20"
                                height="20"
                                fill="currentcolor"
                                fillOpacity={0.7}
                                style={{ marginLeft: 2 }}
                            >
                                <path d="M 4 6 A 1 1 0 0 0 3 7 A 1 1 0 0 0 3 7.0039062 L 3 15 L 3 22.996094 A 1 1 0 0 0 3 23 A 1 1 0 0 0 4 24 A 1 1 0 0 0 4.5917969 23.804688 L 4.59375 23.804688 A 1 1 0 0 0 4.6210938 23.783203 L 15 16.236328 L 15 22.996094 A 1 1 0 0 0 15 23 A 1 1 0 0 0 16 24 A 1 1 0 0 0 16.591797 23.804688 L 16.59375 23.804688 A 1 1 0 0 0 16.621094 23.783203 L 27.541016 15.841797 A 1 1 0 0 0 28 15 A 1 1 0 0 0 27.556641 14.169922 L 16.59375 6.1953125 A 1 1 0 0 0 16 6 A 1 1 0 0 0 15 7 A 1 1 0 0 0 15 7.0039062 L 15 13.765625 L 4.59375 6.1953125 A 1 1 0 0 0 4 6 z" />
                            </svg>
                        </IconButton>
                    </Flex>

                    <Flex align="center" gap="2" p="5">
                        <VolumeNoneIcon color="var(--gray-a9)" />
                        <Slider
                            tabIndex={tabIndex}
                            defaultValue={[80]}
                            variant="soft"
                            color="gray"
                            radius="full"
                            size="2"
                            style={{ width: 80 }}
                        />
                        <VolumeMaxIcon color="var(--gray-a9)" />
                    </Flex>
                </Flex>
            </Box>
        </Card>

    );

};

const songsHistory = [
    {
        name: "Sunday Rain",
        artist: "Foo Fighters",
        album: "Concrete and Gold",
        length: "6:11",
        cover:
            "https://workos.imgix.net/images/28bf3f7c-4ad7-4bd9-9064-c63d2676c8dd.png?auto=format&fit=clip&q=80&w=192",
        color: "hsl(40, 30%, 55%)",
    },
    {
        name: "Left Hand Free",
        artist: "Alt-J",
        album: "This Is All Yours",
        length: "2:54",
        cover:
            "https://workos.imgix.net/images/8d431b64-ebe8-41be-b986-2f59cb5c567d.png?auto=format&fit=clip&q=80&w=192",
        color: "hsl(330, 70%, 64%)",
    },
    {
        name: "Last",
        artist: "Nine Inch Nails",
        album: "Broken",
        length: "4:45",
        cover:
            "https://workos.imgix.net/images/5f495e55-4bac-4573-b97f-bac55d4f3a82.png?auto=format&fit=clip&q=80&w=192",
        color: "hsl(30, 100%, 50%)",
    },
    {
        name: "13LACK 13ALLOONZ (feat. Twelve’len & GoldLink)",
        artist: "Denzel Curry",
        album: "TA13OO",
        length: "3:31",
        cover:
            "https://workos.imgix.net/images/f1b1ff42-eae9-4fcd-9c7f-c3ed92594395.png?auto=format&fit=clip&q=80&w=192",
        color: "hsl(0, 0%, 25%)",
    },
    {
        name: "Self Control",
        artist: "Frank Ocean",
        album: "Blond",
        length: "4:10",
        cover:
            "https://workos.imgix.net/images/419f09bc-99ab-4eae-8e71-d33f0577bd47.png?auto=format&fit=clip&q=80&w=192",
        color: "hsl(80, 20%, 40%)",
    },
    {
        name: "Trippy (feat. J. Cole)",
        artist: "Anderson .Paak",
        album: "Oxnard",
        length: "5:24",
        cover:
            "https://workos.imgix.net/images/daab7042-222f-433f-abcb-15811b8a43da.png?auto=format&fit=clip&q=80&w=192",
        color: "hsl(193, 15%, 45%)",
    },
    {
        name: "Nightclubbing",
        artist: "Iggy Pop",
        album: "The Idiot",
        length: "4:16",
        cover:
            "https://workos.imgix.net/images/85451af7-27bf-4bbb-88e7-088caf762ed5.png?auto=format&fit=clip&q=80&w=192",
        color: "hsl(34, 7%, 45%)",
    },
    {
        name: "Heaven Beside You",
        artist: "Alice in Chains",
        album: "Alice in Chains",
        length: "5:28",
        cover:
            "https://workos.imgix.net/images/72edfcaf-2e5b-492c-bb5b-60a031f001c9.png?auto=format&fit=clip&q=80&w=192",
        color: "hsl(289, 3%, 51%)",
    },
    {
        name: "Night After Night",
        artist: "Laura Marling",
        album: "A Creature I Don’t Know",
        length: "5:08",
        cover:
            "https://workos.imgix.net/images/0cce32ae-6890-419e-b01c-2e89d36cb883.png?auto=format&fit=clip&q=80&w=192",
        color: "hsl(40, 13%, 83%)",
    },
    {
        name: "HEAVN",
        artist: "Jamila Woods",
        album: "HEAVN",
        length: "4:23",
        cover:
            "https://workos.imgix.net/images/e865c892-5cbe-4d1f-b4eb-e2bc301087f0.png?auto=format&fit=clip&q=80&w=192",
        color: "hsl(32, 95%, 67%)",
    },
];


const HeartIcon = (props: React.ComponentPropsWithoutRef<"svg">) => (
    <svg
        xmlns="http://www.w3.org/2000/svg"
        viewBox="0 0 30 30"
        width="20"
        height="20"
        fill="currentcolor"
        {...props}
    >
        <path d="M 9.5449219 3 C 5.3895807 3 2 6.3895806 2 10.544922 C 2 14.283156 4.9005496 18.084723 7.6601562 21.119141 C 10.419763 24.153558 13.171875 26.369141 13.171875 26.369141 A 1.0001 1.0001 0 0 0 13.197266 26.388672 C 13.642797 26.725148 14.201794 26.943857 14.808594 26.984375 A 1.0001 1.0001 0 0 0 15 27 A 1.0001 1.0001 0 0 0 15.189453 26.984375 A 1.0001 1.0001 0 0 0 15.199219 26.982422 C 15.802918 26.940449 16.359155 26.723674 16.802734 26.388672 A 1.0001 1.0001 0 0 0 16.828125 26.369141 C 16.828125 26.369141 19.580237 24.153558 22.339844 21.119141 C 25.099451 18.084722 28 14.283156 28 10.544922 C 28 6.3895806 24.610419 3 20.455078 3 C 17.841043 3 15.989939 4.4385487 15 5.4570312 C 14.010061 4.4385487 12.158957 3 9.5449219 3 z M 9.5449219 5 C 12.276127 5 13.937826 7.2424468 14.103516 7.4746094 A 1.0001 1.0001 0 0 0 14.994141 8.0136719 A 1.0001 1.0001 0 0 0 15.017578 8.0136719 A 1.0001 1.0001 0 0 0 15.041016 8.0117188 A 1.0001 1.0001 0 0 0 15.117188 8.0058594 A 1.0001 1.0001 0 0 0 15.892578 7.4785156 C 16.049938 7.2575052 17.716133 5 20.455078 5 C 23.529737 5 26 7.4702629 26 10.544922 C 26 13.147688 23.499768 16.870104 20.859375 19.773438 C 18.227966 22.666891 15.607768 24.780451 15.589844 24.794922 C 15.414236 24.925626 15.219097 25 15 25 C 14.780903 25 14.585764 24.92563 14.410156 24.794922 C 14.392236 24.780452 11.772034 22.666891 9.140625 19.773438 C 6.5002316 16.870105 4 13.147688 4 10.544922 C 4 7.4702629 6.470263 5 9.5449219 5 z" />
    </svg>
);

const VolumeNoneIcon = (props: React.ComponentPropsWithoutRef<"svg">) => (
    <svg
        xmlns="http://www.w3.org/2000/svg"
        viewBox="0 0 20 20"
        width="20"
        height="20"
        fill="currentcolor"
        fillOpacity={0.7}
        {...props}
    >
        <path d="M16.3333 4.66669L13.5286 7.33335H11C9.89533 7.33335 9 8.22869 9 9.33335V10.6667C9 11.7714 9.89533 12.6667 11 12.6667H13.5286L16.3333 15.3334V4.66669Z" />
    </svg>
);

const VolumeMaxIcon = (props: React.ComponentPropsWithoutRef<"svg">) => (
    <svg
        xmlns="http://www.w3.org/2000/svg"
        viewBox="0 0 30 30"
        width="20"
        height="20"
        fill="currentcolor"
        fillOpacity={0.7}
        {...props}
    >
        <path d="M 20.037109 5.6464844 A 1.0001 1.0001 0 0 0 19.236328 7.2734375 C 20.963426 9.4832305 22 12.243759 22 15.255859 C 22 18.055119 21.105815 20.636923 19.59375 22.763672 A 1.0001 1.0001 0 1 0 21.222656 23.921875 C 22.962591 21.474623 24 18.4826 24 15.255859 C 24 11.78396 22.799402 8.5851757 20.8125 6.0429688 A 1.0001 1.0001 0 0 0 20.037109 5.6464844 z M 11 7 L 6.7929688 11 L 3 11 C 1.343 11 0 12.343 0 14 L 0 16 C 0 17.657 1.343 19 3 19 L 6.7929688 19 L 11 23 L 11 7 z M 14.738281 8.5917969 A 1.0001 1.0001 0 0 0 14.001953 10.291016 C 15.239451 11.587484 16 13.328154 16 15.255859 C 16 16.979025 15.392559 18.553804 14.380859 19.796875 A 1.0001 1.0001 0 1 0 15.931641 21.058594 C 17.219941 19.475665 18 17.450694 18 15.255859 C 18 12.799565 17.023721 10.559688 15.449219 8.9101562 A 1.0001 1.0001 0 0 0 14.738281 8.5917969 z" />
    </svg>
);


export default LiveAnalysisSessionRecording;