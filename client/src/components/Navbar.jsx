import React from 'react'
import { useState } from 'react'
import AnchorLink from 'react-anchor-link-smooth-scroll'
import useMediaQuery from '../hooks/useMediaQuery'
const Link = ({page, selectedPage, setSelectedPage}) => {
    const lowerCasePage = page.toLowerCase(); 
    return (
        <AnchorLink
        className={`${selectedPage === lowerCasePage ? "text-yellow" : ""}
        hover:text-yellow transition duration-500`}
        href={`#${lowerCasePage}`}
        onClick={()=>setSelectedPage(lowerCasePage)}>
            {page}
        </AnchorLink>
    )
}
const Navbar = ({isTopofPage, selectedPage, setSelectedPage}) => {
    const [isMenuToggled, setIsMenuToggled] = useState(false);
    const isAboveSmallScreens = useMediaQuery("(min-width: 768px)")
    const navbarBackGround = isTopofPage ? "" : "bg-red"
    return (
    <nav className={`${navbarBackGround} z-40 w-full fixed top-0 py-6`}>
        <div className='flex items-center justify-between mx-auto w-5/6'>
            <h4 className='font-playfair text-3xl font-bold'>DB</h4>
            {/* Desktop NavBar*/}
            {isAboveSmallScreens ? (
                <div className='flex justify-between gap-16 font-opensans text-sm font-semibold'> 
                    <Link
                    page="form"
                    selectedPage={selectedPage}
                    setSelectedPage={setSelectedPage} />
                    <Link
                    page="Contact"
                    selectedPage={selectedPage}
                    setSelectedPage={setSelectedPage} />
                </div>
            ) :
            (
                <button
                className='rounded-full bg-red p-2'
                onClick={()=>setIsMenuToggled(!isMenuToggled, console.log(isMenuToggled))}>
                    <img src="../assets/menu-icon.svg" alt="mobileMenu" />
                </button>
            )
        }
        {/*Mobile Menu popUp */}
        {!isAboveSmallScreens && isMenuToggled && (
            <div className='fixed right-0 bottom-0 h-full w-[300px] bg-blue'>
                {/**Close Icon */}
                <div className='flex justify-end p-12'>
                    <button onClick={() => setIsMenuToggled(!isMenuToggled)}>
                        <img src="../assets/close-icon.svg" alt="CloseIcon" />
                    </button>
                </div>
                {/** Menu Items */}
                <div className='flex flex-col gap-10 ml-[33%] text-2xl text-deep-blue'>
                    <Link
                    page="form"
                    selectedPage={selectedPage}
                    setSelectedPage={setSelectedPage} />
                    <Link
                    page="Contact"
                    selectedPage={selectedPage}
                    setSelectedPage={setSelectedPage} />
                </div>
            </div>
        )}
        </div>
    </nav>
  )
}

export default Navbar