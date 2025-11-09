import SocialMediaIcons from "./SocialMediaIcons"

const Footer = () => {
  
  return (
    <footer className="h-64 bg-red pt-10">
        <div className="w-5/6 mx-auto">
            <SocialMediaIcons />
            <div className="md:flex justify-center md:justify-between text-center">
                <p className="font-semibold text-yellow text-2xl font-playfair">dibetes type 2 cdss</p>
                <p className="font-playfair text-md text-yellow">HOSEIN SHOOSHI</p>
            </div>
        </div>
    </footer>
  )
}

export default Footer